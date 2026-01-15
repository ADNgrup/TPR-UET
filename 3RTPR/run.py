import os
import sys
import os.path as op
sys.path.append(os.getcwd())
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train, do_inference
from utils.checkpoint import Checkpointer, copy_params
from utils import save_train_configs, load_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator

from utils.comm import get_rank, synchronize
import argparse
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=0):
    print(f"====seed:{seed}====")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def test(cfg, args):
    # cfg = load_train_configs(config_file)

    cfg.training = False
    logger = setup_logger('DANK!1910', save_dir=args.output_dir, if_train=cfg.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(cfg)
    model_a = build_model(cfg, num_classes=num_classes)
    model_b = build_model(cfg, num_classes=num_classes)
    checkpointer_a = Checkpointer(model_a)
    checkpointer_b = Checkpointer(model_b)
    checkpointer_a.load(f=op.join(args.output_dir, 'best-a.pth'))
    checkpointer_b.load(f=op.join(args.output_dir, 'best-b.pth'))
    model_a.to(device)
    model_b.to(device)
    do_inference([model_a, model_b], test_img_loader, test_txt_loader)
    print("\n\n","============"*5, "\n\n", "\t\t Best pair")
    checkpointer_a.load(f=op.join(args.output_dir, 'best-a-bestpair.pth'))
    checkpointer_b.load(f=op.join(args.output_dir, 'best-b-bestpair.pth'))
    model_a.to(device)
    model_b.to(device)
    do_inference([model_a, model_b], test_img_loader, test_txt_loader)
    
    print("\n\n","============"*5, "\n\n", "\t\t Best pair - merged ap-b")
    checkpointer_a.load(f=op.join(args.output_dir, 'best-a.pth'))
    checkpointer_b.load(f=op.join(args.output_dir, 'best-b-bestpair.pth'))
    model_a.to(device)
    model_b.to(device)
    do_inference([model_a, model_b], test_img_loader, test_txt_loader)

    print("\n\n","============"*5, "\n\n", "\t\t Best pair merged a-bp")
    checkpointer_a.load(f=op.join(args.output_dir, 'best-a-bestpair.pth'))
    checkpointer_b.load(f=op.join(args.output_dir, 'best-b.pth'))
    model_a.to(device)
    model_b.to(device)
    do_inference([model_a, model_b], test_img_loader, test_txt_loader)
    
    print("\n\n","============"*5, "\n\n", "\t\t Best pair merged a-ap")
    checkpointer_a.load(f=op.join(args.output_dir, 'best-a-bestpair.pth'))
    checkpointer_b.load(f=op.join(args.output_dir, 'best-a.pth'))
    model_a.to(device)
    model_b.to(device)
    do_inference([model_a, model_b], test_img_loader, test_txt_loader)

    print("\n\n","============"*5, "\n\n", "\t\t Best pair merged b-bp")
    checkpointer_a.load(f=op.join(args.output_dir, 'best-b-bestpair.pth'))
    checkpointer_b.load(f=op.join(args.output_dir, 'best-b.pth'))
    model_a.to(device)
    model_b.to(device)
    do_inference([model_a, model_b], test_img_loader, test_txt_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UET Person search Args")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--name", '-n', type=str, default='DAUET2001', help='name of the model')
    parser.add_argument("--opt", '-o', type=str, default='Adam', help='name of the optimizer', choices=['Adam', 'SGD', 'AdamW'])
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--test", action="store_true")
    #add params to optimize
    parser.add_argument("--l-names", nargs='+', default=[], type=str)
    parser.add_argument("--sampler", default="random", type=str, choices=['random', 'identity'])
    parser.add_argument("--d-names", default="CUHK-PEDES", type=str, choices=["CUHK-PEDES-M",'CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'])
    #Dataloader
    parser.add_argument("--isize", nargs='+', default=[384, 128], type=int)
    parser.add_argument("--tlen",  default=77, type=int)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--saug-text", action="store_true")
    parser.add_argument("--lr", default=1e-5, type=float, help='learning rate')
    parser.add_argument("--lrx", default=1e-3, type=float, help='learning rate')
    #MODEL SETTING
    parser.add_argument("--sratio", type=float, default=0, help='the ratio for token selection mechanism ')
    parser.add_argument("--erpi", type=float, default=0.5, help='the ratio for image random erasing  ')
    parser.add_argument("--erpt", type=float, default=0.2, help='the ratio for text random erasing  ')
    parser.add_argument("--fusedim", type=int, default=1024, help='the length of fused feature dimension ')
        
    #CCD
    parser.add_argument("--single", action="store_true", help= "using Confident Consensus Division")
    
    parser.add_argument("--ccd", action="store_true", help= "using Confident Consensus Division")
    parser.add_argument("--ccd-r", action='store_true', help= "remake dataset after ccd")
    parser.add_argument("--ccd-a", action='store_true', help= "use all model or randome ")
    parser.add_argument("--ldynamic", action='store_true', help= "enable dynamic model for loss")
    parser.add_argument("--ldynamic-t", type=int, default=0, help= "dynamic model")
    parser.add_argument("--ldynamic-m", type=str, default='min', choices=['min', 'mean', '25', '75', '30'], help= "dynamic model")
    #APL
    parser.add_argument("--noisy_rate", type=float, default=0, choices=[0.0, 0.2,0.5,0.8])
    parser.add_argument("--noisy_file", type=str, default="")

    parser.add_argument("--dev", action="store_true") 
    #TRL
    parser.add_argument("--l-trl-type", type=str, default="soft", choices=['soft', 'hard'])
    parser.add_argument("--l-trl-topk", type=float, default=None)
    parser.add_argument("--l-trl-M", type=float, default=None)
    parser.add_argument("--l-trl-tau", type=float, default=None) 
    #MLM
    parser.add_argument("--l-mlm-prob", type=float, default=None)
    #MAM
    parser.add_argument("--l-mam-prob", type=float, default=None)
    #MGM
    parser.add_argument("--l-mgm-prob", type=float, default=0)
    

    #MMM
    parser.add_argument("--mmm-crossmodel-depth", type=int, default=4, choices=[2, 4, 8, 12, 16, 20])

    parser.add_argument("--lossweight-sdm", type=float, default=0)
    parser.add_argument("--lossweight-mam", type=float, default=0)
    parser.add_argument("--lossweight-trl", type=float, default=0)
   
    
    args = parser.parse_args()

    print("Train for ", args.name)
    cfg = OmegaConf.load(args.cfg)
    import random
    misc = random.randint(1000, 10000)
    set_seed(args.seed)
    cfg.losses.loss_names = args.l_names
    cfg.name = args.name
    cfg.dataloader.sampler = args.sampler
    cfg.dataloader.dataset_name = args.d_names
    cfg.dataloader.batch_size = args.bs
    cfg.trainer.optimizer.opt = args.opt
    cfg.trainer.optimizer.lr = args.lr
    cfg.trainer.optimizer.lrx = args.lrx
    cfg.image_encoder.img_size = args.isize
    cfg.dataloader.text_length = args.tlen
    cfg.erpi = args.erpi
    cfg.erpt = args.erpt
    cfg.dev = args.dev
    print(f'[!!!] USING Random Esrasing for I= {cfg.erpi}, T={cfg.erpt}')
    cfg.noisy_rate = args.noisy_rate
    cfg.noisy_file = args.noisy_file
    print(f'[!!!] USING noisy data with rate as  {cfg.noisy_rate} from {cfg.noisy_file}')
    
    if args.ccd:
        print('[!!!] USING Confident Consensus Division with mode-remake=', args.ccd_r, " and use all model = ", args.ccd_a)
        cfg.ccd.enable = True
        cfg.ccd.remakeds = args.ccd_r
        cfg.ccd.all_model = args.ccd_a
    cfg.image_encoder.local_branch.selection_ratio = args.sratio
    if args.sratio>0:
        print('[!!!] USING Fused feature with sratio = ', args.sratio, "and dim=", args.fusedim)
        cfg.image_encoder.local_branch.dim = args.fusedim

    cfg.losses.dynamic = args.ldynamic; 
    if cfg.losses.dynamic : 
        print(f"[!!!] USING Dynamic mode {args.ldynamic_t} for loss calculation with muy set by {args.ldynamic_m}")
        cfg.losses.dynamic_type = args.ldynamic_t
        cfg.losses.dynamic_muy = args.ldynamic_m
    

    if not args.l_mlm_prob is None:
        print("[!!!]change MLM Prob to ", args.l_mlm_prob)
        cfg.losses.mmm.mlm.mask_prob = args.l_mlm_prob
    if 'mam' in args.l_names:
        print("[!!!] USE MAM with prob =", args.l_mam_prob)
        cfg.losses.mmm.mam.mask_prob = args.l_mam_prob
    if args.l_mgm_prob > 0:
        print("[!!!] USE MGM with prob =", args.l_mgm_prob)
        cfg.losses.mmm.mgm.mask_prob = args.l_mgm_prob
    cfg.losses.mmm.cross_modal.cmt_depth  = args.mmm_crossmodel_depth

    if 'trl' in args.l_names:
        cfg.losses.trl.type = args.l_trl_type;  print("[!!!] USE TRL with type=",  args.l_trl_type)
        if not args.l_trl_topk is None:
            print("\t\t===> TRL.topk is set to ", args.l_trl_topk);cfg.losses.trl.topk = args.l_trl_topk
        if not args.l_trl_M is None:
            print("\t\t===> TRL.margin is set to ", args.l_trl_M);cfg.losses.trl.margin = args.l_trl_M
        if not args.l_trl_tau is None:
            print("\t\t===> TRL.tau is set to ", args.l_trl_tau);cfg.losses.trl.tau = args.l_trl_tau


    cfg.losses.sdm_loss_weight = args.lossweight_sdm
    cfg.losses.trl_loss_weight = args.lossweight_trl
    print("=======LOSS WEIGHT=======")
    for k, v in cfg.losses.items():
        if "loss" in k or "local_branch" in k:
            if isinstance(cfg.losses[k], dict):
                for k_, v_ in cfg.losses[k].items(): print(f"===>{k_}: {v_}")
            else: print(f"\t\t===>{k}: {v}")

    ######
    set_seed(args.seed)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.distributed = num_gpus > 1


    if args.test: test(cfg, args)
    else: #TRAINING
        if cfg.distributed:
            torch.cuda.set_device(cfg.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()

        device = "cuda"
        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        cfg.output_dir = output_dir = op.join(cfg.iocfg.savedir, cfg.dataloader.dataset_name, f'{cur_time}_{cfg.name}_{misc}')
        logger = setup_logger('DANK!1910', save_dir=output_dir, if_train=True, distributed_rank=get_rank())
        logger.info("Using {} GPUs".format(num_gpus))
        save_train_configs(output_dir, cfg)

        # get image-text pair datasets dataloader
        train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(cfg)

        # Build models

        model_a = build_model(cfg, num_classes, "a")
        model_b = build_model(cfg, num_classes, "b")

        logger.info('Total params: %2.fM' % (sum(p.numel() for p in model_a.parameters()) / 1000000.0))

        model_a =model_a.to(device)
        model_b =model_b.to(device)

        optimizer_a = build_optimizer(cfg, model_a)
        optimizer_b = build_optimizer(cfg, model_b)
        scheduler_a = build_lr_scheduler(cfg, optimizer_a)
        scheduler_b = build_lr_scheduler(cfg, optimizer_b)

        is_master = get_rank() == 0
        checkpointer_a = Checkpointer(model_a, optimizer_a, scheduler_a, output_dir, True)
        checkpointer_b = Checkpointer(model_b, optimizer_b, scheduler_b, output_dir, True)
        evaluator = Evaluator(val_img_loader, val_txt_loader)
        checkpointer_b.save("hege")
        checkpointer_a.save("hege", ema=True)
        # raise 
        start_epoch = 1
        if args.single:
            M = [model_a]
            O = [optimizer_a]
            S = [scheduler_a]
            C = [checkpointer_a]       
        else:
            M = [model_a, model_b]
            O = [optimizer_a, optimizer_b]
            S = [scheduler_a, scheduler_b]
            C = [checkpointer_a, checkpointer_b]
        do_train(start_epoch, cfg, M, train_loader, evaluator, O, S, C )
       