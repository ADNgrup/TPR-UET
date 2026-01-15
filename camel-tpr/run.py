import os
import os.path as op
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def test(config_file):
    args = load_train_configs(config_file)

    args.training = False
    logger = setup_logger('DANK!1910', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UET Person search Args")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--test", action="store_true")
    #add params to optimize
    parser.add_argument("--l-names", nargs='+', default=[], type=str)
    parser.add_argument("--sampler", default="random", type=str, choices=['random', 'identity'])
    parser.add_argument("--d-names", default="CUHK-PEDES", type=str, choices=["CUHK-PEDES-M",'CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'])
    #Dataloader
    parser.add_argument("--isize", nargs='+', default=[384, 128], type=int)
    parser.add_argument("--tlen",  default=77, type=int)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--saug", action="store_true")
    parser.add_argument("--saug-text", action="store_true")
    parser.add_argument("--lr", default=1e-5, type=float, help='learning rate')
    #MODEL SETTING
    parser.add_argument("--local-branch", action="store_true")
    parser.add_argument("--sratio", type=float, default=0.5, help='the ratio for token selection mechanism ')
        
    #CCD
    parser.add_argument("--ccd", action="store_true", help= "using Confident Consensus Division")
    #tal
    parser.add_argument("--l-tal-topk", type=float, default=None)
    parser.add_argument("--l-tal-M", type=float, default=None)
    parser.add_argument("--l-tal-tau", type=float, default=None) 
    parser.add_argument("--l-tal-hard", action="store_true") 

    #MLM
    parser.add_argument("--l-mlm-prob", type=float, default=None)
    #MMM
    parser.add_argument("--mmm-crossmodel-depth", type=int, default=4, choices=[2, 4, 8, 12, 16, 20])

    parser.add_argument("--lossweight-sdm", type=float, default=0)
    parser.add_argument("--lossweight-mlm", type=float, default=0)
    parser.add_argument("--lossweight-tal", type=float, default=0)

    parser.add_argument("--lossweight-sdm-local", type=float, default=0)
    parser.add_argument("--lossweight-tal-local", type=float, default=0)

    

    args = parser.parse_args()
    if args.test: test(args.cfg)
    else: #TRAINING
        cfg = OmegaConf.load(args.cfg)
        set_seed(123)
        if len(args.l_names)    > 0:
            print("loss is use = ", args.l_names)
            cfg.losses.loss_names = args.l_names
        cfg.losses.mmm.mlm.use_trick = args.l_mlm_use_trick
        if args.saug_text: print("[!!!] USE MASKTEXT AS caption_ids")
        cfg.dataloader.use_masked_text = args.saug_text
        cfg.dataloader.sampler = args.sampler
        cfg.dataloader.dataset_name = args.d_names
        cfg.dataloader.batch_size = args.bs
        cfg.dataloader.strong_aug = args.saug
        cfg.trainer.optimizerlr = args.lr
        cfg.image_encoder.img_size = args.isize
        print("Image size is set to = ", args.isize)
        cfg.dataloader.text_length = args.tlen
        print("TextLength is set to = ", args.tlen)
        if args.ccd:
            print('[!!!] USING Confident Consensus Division')
            cfg.ccd.enable = True            
        if args.local_branch:
            print('[!!!] Integrate token Selection branch with Selection ratio =', args.sratio)
            cfg.image_encoder.local_branch.enable = args.local_branch
            cfg.image_encoder.local_branch.selection_ratio = args.sratio
            if args.lossweight_sdm_local > 0 :
                print("\t\t[!!!]change \\lamda of SDM(L) to", args.lossweight_sdm_local)
                cfg.losses.local_branch.sdm_loss_weight = args.lossweight_sdm_local
            if  args.lossweight_tal_local > 0 :      
                print("\t\t[!!!]change \\lamda of tal(L) to ", args.lossweight_tal_local)
                cfg.losses.local_branch.tal_loss_weight = args.lossweight_tal_local

        if not args.l_mlm_prob is None:
            print("[!!!]change MLM Prob to ", args.l_mlm_prob)
            cfg.losses.mmm.mlm.mask_prob = args.l_mlm_prob
        if 'mlm' in args.l_names:
            if args.l_mlm_hmode: 
                print(f"\t\t===>change MLM to HMODE | \t-->will select patchs having attention belong the rank lower {args.l_mlm_hmode_limit} %")
                cfg.losses.mmm.mlm.hard_mode = args.l_mlm_hmode
                cfg.losses.mmm.mlm.hard_mode_limit = args.l_mlm_hmode_limit
        
        cfg.losses.mmm.cross_modal.cmt_depth  = args.mmm_crossmodel_depth


        if not args.l_tal_topk is None:
            print("[!!!] tal.topk is set to ", args.l_tal_topk)
            cfg.losses.tal.topk = args.l_tal_topk
        if not args.l_tal_M is None:
            print("[!!!] tal.margin is set to ", args.l_tal_M)
            cfg.losses.tal.margin = args.l_tal_M
        if not args.l_tal_tau is None:
            print("[!!!] tal.tau is set to ", args.l_tal_tau)
            cfg.losses.tal.tau = args.l_tal_tau


        if not args.lossweight_mim is None:
            print("[!!!]change \\lamda of MIM ")
            cfg.losses.mim_loss_weight = args.lossweight_mim
        if not args.lossweight_sdm is None:
            print("[!!!]change \\lamda of SDM to  ", args.lossweight_sdm)
            cfg.losses.sdm_loss_weight = args.lossweight_sdm
        if not args.lossweight_tal is None:
            print("[!!!]change \\lamda of tal to ", args.lossweight_tal)
            cfg.losses.tal_loss_weight = args.lossweight_tal

        print("=======LOSS WEIGHT=======")
        for k, v in cfg.losses.items():
            if "loss" in k or "local_branch" in k:
                if isinstance(cfg.losses[k], dict):
                    for k_, v_ in cfg.losses[k].items(): print(f"===>{k_}: {v_}")
                else: print(f"\t\t===>{k}: {v}")

        ######
        set_seed(1)
        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        cfg.distributed = num_gpus > 1

        if cfg.distributed:
            torch.cuda.set_device(cfg.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()

        device = "cuda"
        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        import random
        misc = random.randint(1000, 10000)
        cfg.output_dir = output_dir = op.join(cfg.iocfg.savedir, cfg.dataloader.dataset_name, f'{cur_time}_{cfg.name}_{misc}')
        logger = setup_logger('DANK!1910', save_dir=output_dir, if_train=True, distributed_rank=get_rank())
        logger.info("Using {} GPUs".format(num_gpus))
        # save_train_configs(output_dir, cfg)

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

        start_epoch = 1

        do_train(start_epoch, cfg, [model_a, model_b], train_loader, evaluator, [optimizer_a, optimizer_b], [scheduler_a, scheduler_b], [checkpointer_a, checkpointer_b])