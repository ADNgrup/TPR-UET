import torch
import numpy as np
from typing import List 
from model.build import DATPS
from losses import objectives
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import random
def _split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)


def _compute_per_loss_4ccd(args, models:List[DATPS], batch, **kwargs ):    
    loss_values = [ 0 for _ in models]
    Gsims        = 0
    for midx, model in enumerate(models):
        #<-------DONT CARE THIS PART---->
        org_tasks = model.args.losses.loss_names
        org_name  = model.name 
        model.name = random.choice([i for i in ['a', 'b'] if i != org_name])
        model.args.losses.loss_names = [k for k in org_tasks if not k in ['mim', 'mlm']]
        model_output    = model(batch)
        model.args.losses.loss_names = org_tasks
        model.name = org_name
        #<---DONT CARE THE PART ABOVE--->
        logit_scale     = model_output['logit_scale']
        gInorm_feats    = model_output["image_norms_fused_feats"] #local feature
        gTnorm_feats    = model_output["text_norms_fused_feats"]
        gscoret2i       = gTnorm_feats @ gInorm_feats.t()
        cur_task  = args.losses.loss_names
        Gsims                         += gscoret2i.diagonal().detach().cpu()
        Gloss = 0
        if 'sdm' in cur_task and args.losses.sdm_loss_weight > 0: Gloss += objectives.compute_sdm(gscoret2i, batch['pids'], logit_scale) * args.losses.sdm_loss_weight
        elif 'trl' in cur_task and args.losses.trl_loss_weight > 0: Gloss += objectives.compute_TRL(gscoret2i, batch['pids'], margin=args.losses.trl.margin, tau=args.losses.trl.tau) * args.losses.trl_loss_weight

        loss_values[midx] += Gloss      
    return [loss.detach().cpu() for loss in loss_values], (Gsims / len(models)).detach().cpu()


def get_per_sample_loss(args, models, data_loader):
    print("\t\t\t===================Calculate Consensus Division before starting epoch===================")
    for model in models: model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()
    check = True
    pids, lossA, lossB = torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size)
    Sims = torch.zeros(data_size)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            with torch.no_grad(): 
                glosses, gsims = _compute_per_loss_4ccd(args, models, batch)
                if len(glosses) == 1:
                    glosses = [glosses[0], glosses[0]]
                glossa, glossb = glosses
                for b in range(glossa.size(0)):
                    lossA[index[b]]= glossa[b]
                    lossB[index[b]]= glossb[b]
                    Sims[index[b]] = gsims[b]

    losses_A = ((lossA-lossA.min())/(lossA.max()-lossA.min() + 1e-9)).reshape(-1,1)  
    losses_B = ((lossB-lossB.min())/(lossB.max()-lossB.min() + 1e-9)).reshape(-1,1) 
    print('\n\t\t\t===================Fitting GMM ...===================') 
 
    if  model.args.dataloader.dataset_name=='RSTPReid':
        # should have a better fit 
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
        gmm_B = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    gmm_A.fit(losses_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(losses_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]
    gmm_B.fit(losses_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(losses_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]
 
 
    pred_A = _split_prob(prob_A, 0.5)
    pred_B = _split_prob(prob_B, 0.5)
  
    return torch.Tensor(pred_A), torch.Tensor(pred_B), torch.Tensor(Sims)
