import logging
import time
import torch
import math
import numpy as np
from losses import objectives
from losses import ema_loss
from model.build import DATPS
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
# from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torchvision.transforms as T
import torch.nn.functional as F  
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from typing import List

def _mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices=[49407, 49408, 49406],
            mlm_probability=0.15, replace_prob=0.2, orginal_prob=0.2, ignore_index=0, probability_matrix=None):

    device = inputs.device
    labels = inputs.clone()

    # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
    if probability_matrix is None:
        probability_matrix = torch.full(labels.shape, mlm_probability, device=device) * (inputs  != 0).long()
    special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
    for sp_id in special_token_indices:
        special_tokens_mask = special_tokens_mask | (inputs==sp_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0) #unmasked special tokens
    mlm_mask = torch.bernoulli(probability_matrix).bool()
    labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

    # mask  (mlm_probability * (1-replace_prob-orginal_prob))
    mask_prob = 1 - replace_prob - orginal_prob # 0.8
    mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
    inputs[mask_token_mask] = mask_token_index

    # replace with a random token (mlm_probability * replace_prob)
    rep_prob = replace_prob / (replace_prob + orginal_prob)
    replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    inputs[replace_token_mask] = random_words[replace_token_mask]


    # do nothing (mlm_probability * orginal_prob)
    pass

    return inputs, labels, mlm_mask

def _split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)


def _compute_per_loss_4ccd(args, models:List[DATPS], batch, **kwargs ):    
    gscoret2i =lscoret2i = 0
    for model in models:
        #<-------DONT CARE THIS PART---->
        org_tasks = model.args.losses.loss_names
        model.args.losses.loss_names = [k for k in org_tasks if not k in ['mim', 'mlm']]
        model_output    = model(batch)
        model.args.losses.loss_names = org_tasks

        # model_output    = model(batch)
        logit_scale     = model_output['logit_scale']
        lInorm_feats    = model_output["image_norms_selected"] #local feature
        lTnorm_feats    = model_output["text_norms_selected"]
        gInorm_feats    = model_output['gimage_norms']
        gTnorm_feats    = model_output['gtext_norms']
        gscoret2i       += gTnorm_feats @ gInorm_feats.t()
        if not lInorm_feats is None: 
            lscoret2i       += lTnorm_feats @ lInorm_feats.t() 
    gscoret2i /= len(models)
    lscoret2i /= len(models)

    Gloss = Lloss = 0
    cur_task  = args.losses.loss_names
    if 'sdm' in cur_task:
        _gsdm = objectives.compute_sdm(gscoret2i, batch['pids'], logit_scale) * args.losses.sdm_loss_weight
        Gloss += _gsdm 

    if 'tal' in cur_task:
        _gtal  = objectives.compute_tal(gscoret2i, batch['pids'],  tau=args.losses.tal.tau, margin=args.losses.tal.margin) * args.losses.tal_loss_weight
        Gloss += _gtal 
    
    if args.image_encoder.local_branch.enable:
        if 'sdm' in cur_task and args.losses.local_branch.sdm_loss_weight > 0:
            _lsdm = objectives.compute_sdm(lscoret2i, batch['pids'], logit_scale) * args.losses.local_branch.sdm_loss_weight
            Lloss += _lsdm 

        if 'tal' in cur_task and args.losses.local_branch.tal_loss_weight > 0:
            _ltal = objectives.compute_tal(lscoret2i, batch['pids'], tau=args.losses.tal.tau, margin=args.losses.tal.margin) * args.losses.local_branch.tal_loss_weight 
            Lloss += _ltal 
    else : Lloss = 0

    return Gloss.detach().cpu() , None if isinstance(Lloss, int) else Lloss.detach().cpu()


def get_loss(args, models, data_loader):
    print("Calculate Consensus Division before starting epoch")
    for model in models: model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()
    check = True
    pids, lossG, lossL = torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size)
    ifeatsG, ifeatsL   = torch.zeros((data_size, 512)), torch.zeros((data_size, 512))
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            with torch.no_grad(): 
                gloss, lloss = _compute_per_loss_4ccd(args, models, batch)
                if lloss is None: 
                    lloss = gloss.clone() #if we dont use local branch
                for b in range(gloss.size(0)):
                    lossG[index[b]]= gloss[b]
                    lossL[index[b]]= lloss[b]
                if i % 250 == 0:
                    print(f'\t==>compute loss batch {i}')
    losses_G = (lossG-lossG.min())/(lossG.max()-lossG.min() + 1e-9)    
    losses_L = (lossL-lossL.min())/(lossL.max()-lossL.min() + 1e-9)

    input_loss_G = losses_G.reshape(-1,1) 
    input_loss_L = losses_L.reshape(-1,1)
 
    print('\nFitting GMM ...') 
 
    if  model.args.dataloader.dataset_name=='RSTPReid':
        # should have a better fit 
        gmm_G = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
        gmm_L = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_G = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_L = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    gmm_G.fit(input_loss_G.cpu().numpy())
    prob_G = gmm_G.predict_proba(input_loss_G.cpu().numpy())
    prob_G = prob_G[:, gmm_G.means_.argmin()]
    gmm_L.fit(input_loss_L.cpu().numpy())
    prob_L = gmm_L.predict_proba(input_loss_L.cpu().numpy())
    prob_L = prob_L[:, gmm_L.means_.argmin()]
 
 
    pred_G = _split_prob(prob_G, 0.5)
    pred_L = _split_prob(prob_L, 0.5)
  
    return torch.Tensor(pred_G), torch.Tensor(pred_L), ifeatsG, ifeatsL, pids

def do_train(start_epoch, args, models:List[DATPS], train_loader, evaluator, optimizers:list, schedulers:list, checkpointers:list):

    log_period = args.trainer.log_period
    eval_period = args.trainer.eval_period
    device = "cuda"
    num_epoch = args.trainer.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("DANK!1910.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        'tal_loss': AverageMeter(),
        "mlm_loss": AverageMeter(),

        "sdm_loss(L)": AverageMeter(),
        "tal_loss(L)": AverageMeter(),

    }


    best_top1 = 0.0
    args.cur_step = 0
    stpe = len(train_loader)  #step per epoch
    total_step =  args.total_step = num_epoch * stpe
    current_task = args.losses.loss_names
    logger.info(f'Training Model with {current_task} tasks')

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values(): meter.reset()
        #Calculate the corresponsive consensus division before start training for dataset with N-pairs
        if args.ccd.enable:
            pred_G, pred_L, _, _, _ = get_loss(args, models, train_loader)
            consensus_division = pred_G + pred_L # 0,1,2 
            label_hat = consensus_division.clone() #Nx1
            consensus_division[consensus_division==1] += torch.randint(0, 2, size=(((consensus_division==1)+0).sum(),)) 
            label_hat[consensus_division>1] = 1  
            label_hat[consensus_division<=1] = 0  
            print("\t\t====Number correct pairs:", (label_hat / (label_hat + 1e-8) ).sum(), "over", len(label_hat))
        else:
            label_hat = torch.ones((train_loader.dataset.__len__()))
        
        for model in models: model.train()
        for n_iter, samples in enumerate(train_loader):
            rets = {model.name : dict() for model in models}
            cur_step = args.cur_step =  (epoch-1) * stpe + n_iter + 1
            samples['label_hat'] = label_hat[samples['index']]
            samples = {k: v.to(device) for k, v in samples.items()}
            # ##########################################
            with torch.no_grad():
                if 'mlm' in current_task:
                    mlm_prob = args.losses.mmm.mlm.mask_prob
                    for model in models: 
                        mlm_ids, mlm_labels, mlm_mask   = _mask_tokens(samples['caption_ids'].clone(), mask_token_index=train_loader.dataset.tokenizer.encoder["<|mask|>"], 
                                                            vocab_size=args.text_encoder.vocab_size-3,  mlm_probability=mlm_prob,  ignore_index=0, probability_matrix=None)
                        samples[f'mlm_ids_{model.name}'], samples[f'mlm_labels_{model.name}'] = mlm_ids, mlm_labels
                        
            #Student Encoding
            logit_scale = 0
            for midx, model in enumerate(models):
                batch = {k: v.to(device) for k, v in samples.items()}
                model_output    = model(batch)
                logit_scale     = model_output['logit_scale']
                gInorm_feats    = model_output['gimage_norms']      #global feature
                gTnorm_feats    = model_output['gtext_norms']
                lInorm_feats    = model_output["image_norms_selected"] #local feature
                lTnorm_feats    = model_output["text_norms_selected"]
                gscoret2i = gTnorm_feats @ gInorm_feats.t()
                rets[model.name].update({'temperature': 1 / logit_scale})
                
                if 'sdm' in current_task:
                    _loss_value = objectives.compute_sdm(gscoret2i, batch['pids'], logit_scale) *  batch['label_hat'] #==> only take the sample with 1 label
                    _loss_value = _loss_value.sum() / (batch['label_hat'] .sum() + 1e-8)  * args.losses.sdm_loss_weight
                    rets[model.name].update({'sdm_loss': _loss_value })
                if 'tal' in current_task:
                    _loss_value = objectives.compute_tal(gscoret2i, batch['pids'], tau=args.losses.tal.tau, margin=args.losses.tal.margin ) *  batch['label_hat']  #==> only take the sample with 1 label
                    _loss_value = _loss_value.sum() * args.losses.tal_loss_weight #TAL is so small so should keep sum value
                    rets[model.name].update({'tal_loss':_loss_value  })
                    

                if args.image_encoder.local_branch.enable:
                    lscoret2i = lTnorm_feats @ lInorm_feats.t()
                    if 'sdm' in current_task and args.losses.local_branch.sdm_loss_weight > 0:
                        _loss_value = objectives.compute_sdm(lscoret2i, batch['pids'], logit_scale) * batch['label_hat'] #==> only take the sample with 1 label
                        _loss_value = _loss_value.sum() / (batch['label_hat'] .sum() + 1e-8) * args.losses.local_branch.sdm_loss_weight
                        rets[model.name].update({'sdm(L)_loss':_loss_value })
                    if 'tal' in current_task:
                        _loss_value = objectives.compute_tal(lscoret2i, batch['pids'],  tau=args.losses.tal.tau, margin=args.losses.tal.margin, topk=args.losses.tal.topk) * batch['label_hat'] 
                        _loss_value = _loss_value.sum() * args.losses.local_branch.tal_loss_weight
                        rets[model.name].update({'tal(L)_loss': _loss_value  })

                if 'mlm' in current_task:
                    mlm_logits      = model_output['mlm_logits']
                    mlm_labels      = batch[f'mlm_labels_{model.name}']
                    for i in range(mlm_labels.shape[0]):
                        mlm_labels[i] = (batch['label_hat'][i].float() * mlm_labels[i].float()) 
                    mlm_labels      = mlm_labels.reshape(-1)
                    mlm_logits = mlm_logits.reshape(-1, args.text_encoder.vocab_size)
                    mlm_loss   =  objectives.compute_mlm(mlm_logits, mlm_labels, reducation='none') 
                    mlm_loss   = mlm_loss.sum() / ((mlm_labels > 0).float().sum() + 1e-6)  *  args.losses.mlm_loss_weight
                    rets[model.name].update({'mlm_loss': mlm_loss})

            #########################################
                total_loss = sum([v for k, v in rets[model.name].items() if "loss" in k])
                optimizers[midx].zero_grad()
                total_loss.backward()
                optimizers[midx].step()


            # LOG synchronize()
            ret = dict()
            for k, v in rets["a"].items(): ret[k] = 0
            for k, v in rets.items(): 
                for kk, vv in v.items(): ret[kk] += vv
            for k, v in ret.items(): ret[k] /= len(rets)
    
            batch_size = batch['images_a'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['tal_loss'].update(ret.get('tal_loss', 0), batch_size)

            meters['sdm_loss(L)'].update(ret.get('sdm(L)_loss', 0), batch_size)
            meters['tal_loss(L)'].update(ret.get('tal(L)_loss', 0), batch_size)

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if (v.avg > 0):
                        info_str += f", {k}: {v.avg:.3f}"
                # info_str += f", Base Lr: {scheduler.get_lr()[0]:.3e}"
                logger.info(info_str)

        for scheduler in schedulers: scheduler.step()

        #use tensorboard to log
        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        # for k, v in meters.items():
        #     if v.avg > 0:
        #         tb_writer.add_scalar(k, v.avg, epoch)

        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,train_loader.batch_size / time_per_batch))
        if epoch % eval_period  == 0 :
            if get_rank() == 0:
                for midx, model in enumerate(models):
                    top1, table = evaluator.eval([model.eval()], i2t_metric=False, return_table=True, print_log=False)

                    torch.cuda.empty_cache()
                    if best_top1 < top1:
                        best_top1 = top1
                        arguments["epoch"] = epoch
                        checkpointers[midx].save("best", **arguments)
                    logger.info("\n\t==MODEL {}=> Validation Single Results - Epoch: {} - Top1={} \n ".format(model.name, epoch, top1) + str(table) + "===============")

                top1, table = evaluator.eval([model.eval() for model in models], i2t_metric=False, return_table=True, print_log=False)
                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    for checkpointer in checkpointers: checkpointer.save("best-ensemble", **arguments)
                logger.info("\n\t==MODEL MEAN=> Validation Ensemble Results - Epoch: {} - Top1={} \n ".format(epoch, top1) + str(table) + "===============")

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("DANK!1910.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
