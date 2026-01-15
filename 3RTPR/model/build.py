from losses import objectives
from losses import ema_loss
from model.clip_model import QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import model.layer as LocalLayer
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math

    
class _ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer4crossmodule(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[_ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class DATPS(nn.Module):
    def __init__(self, args, num_classes=11003, name="a"):
        super().__init__()
        self.args = args
        self.name = name
        self.num_classes = num_classes
        self.use_token_selection = self.args.image_encoder.local_branch.enable

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.image_encoder.name, args.image_encoder.img_size, args.image_encoder.stride_size, download_root=args.iocfg.datadir)
        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.base_model.visual.conv1.named_parameters():
            v.requires_grad_(False)


        self.embed_dim = self.cls_embed_dim = base_cfg['embed_dim']
        self.sratio =  self.args.image_encoder.local_branch.selection_ratio
        self.vtselection = LocalLayer.VisualFusedEmbeddingLayer(input_dim=768, embed_dim=args.image_encoder.local_branch.dim, ratio=self.sratio)
        self.ttselection = LocalLayer.TexualFusedEmbeddingLayer(input_dim=512, embed_dim=args.image_encoder.local_branch.dim, ratio=self.sratio)


        
        self.logit_scale = torch.ones([]) * (1 / args.image_encoder.temperature)

        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)
        self.vision_patch_size = base_cfg['vision_patch_size']

        if len({"mam","mim", 'ntlm', 'apl'} & set(args.losses.loss_names)) > 0:
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,self.embed_dim // 64, batch_first=True)
            self.cross_modal_transformer = Transformer4crossmodule(width=self.embed_dim, layers=args.losses.mmm.cross_modal.cmt_depth, heads=self.embed_dim // 64)
            scale = self.cross_modal_transformer.width**-0.5
            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)


            if 'mam' in args.losses.loss_names :
                self.mam_head = nn.Sequential(
                    OrderedDict([
                                # ('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                                # ('gelu', QuickGELU()),
                                ('ln', LayerNorm(self.embed_dim)),
                                ('fc', nn.Linear(self.embed_dim, args.text_encoder.vocab_size))]))
                # nn.init.normal_(self.mam_head.dense.weight, std=fc_std)
                nn.init.normal_(self.mam_head.fc.weight, std=proj_std)

    #######################################    METHOD SECTION    ####################################################

    def cross_former(self, q, k, v, **kwargs):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]

        x = x.permute(1, 0, 2)  # NumxLengthxDim -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        return x


    def encode_image(self, image, att_ret=False):
        x, att, org_x = self.base_model.encode_image(image)
        if att_ret: 
            return x, att
        x = x[:, 0, :].float()
        return x
    def encode_text(self, text, att_ret=False):
        x, att, org_x = self.base_model.encode_text(text.long())
        if att_ret: 
            return x, att
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        return x
    
    def encode_image_fuse(self, image):
        x,atten_i, org_x = self.base_model.encode_image(image)
        x, _  = self.vtselection(x, org_x, atten_i) 
        return x.float()
    def encode_text_fuse(self, text):
        x,atten_t, org_x = self.base_model.encode_text(text.long())
        x, _ = self.ttselection(x, org_x, text.long(), atten_t)
        return x.float()


    ###MAIN Forward function
    def forward(self, batch):
        #data parse
        images = batch[f'images_{self.name}']
        caption_ids = batch['caption_ids']
        #text augmented input
        if self.args.erpt > 0: caption_ids = batch[f'masked_caption_ids_{self.name}']
            
        #Encode
        #//G
        image_feats, text_feats, image_attscore, text_attscore, image_rfeatures, text_rfeatures = self.base_model(images, caption_ids) #torch.Size([B, tokens, 512]) torch.Size([1, tokens, 512])
        image_fused_feats, li_feats = self.vtselection(image_feats, image_rfeatures, image_attscore)
        text_fused_feats,  lt_feats = self.ttselection(text_feats, text_rfeatures, caption_ids, text_attscore)
           
        logit_scale = self.logit_scale
        if "mam" in self.args.losses.loss_names:
            mam_text = batch[f'masked_att_ids_{self.name}']
            mam_feats, _ = self.base_model.encode_text(mam_text)
            mam_fuse_feats = self.cross_former(mam_feats, image_feats, image_feats)  #BxPatchsxD
            mam_logits = self.mam_head(mam_fuse_feats)   #BxPatchsxD
        else: mam_logits = None

        return {
            "logit_scale": logit_scale,
            "image_norms_fused_feats" : image_fused_feats / image_fused_feats.norm(dim=-1, keepdim=True), #if self.use_token_selection else None,
            "text_norms_fused_feats" : text_fused_feats / text_fused_feats.norm(dim=-1, keepdim=True),   #if self.use_token_selection else None,
            "mam_logits": mam_logits,
            "":None
        }


def build_model(args, num_classes=11003, name='a'):
    model = DATPS(args, num_classes, name)
    # covert model to fp16
    convert_weights(model)
    return model