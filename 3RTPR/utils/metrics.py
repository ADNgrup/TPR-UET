from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from model.build import DATPS

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.to(g_pids.device)]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("DANK!1910.eval")

    def _compute_embedding(self, model:DATPS):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids

    def _compute_embedding_local(self, model:DATPS):
        model = model.eval() 
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text_fuse(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image_fuse(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0) 
        return qfeats, gfeats, qids, gids

    def _compute_metrics(self, qfeats, gfeats, qids, gids, i2t_metric=False):
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        return table, t2i_cmc
    
    def get_metrics(self, similarity, qids, gids, n_, retur_indices=False):
        t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        if retur_indices:
            return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, indices]
        else:
            return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP]


    def eval(self, models, i2t_metric=False, return_table=False, print_log=True, return_all=False, sims_dict=None, qids=None, gids=None): 
        sims_global = sims_local = 0
        if sims_dict is None:
            for model in models:
                qfeats, gfeats, qids, gids = self._compute_embedding(model)
                qfeats = F.normalize(qfeats, p=2, dim=1) # text features
                gfeats = F.normalize(gfeats, p=2, dim=1) # image features
                sims_global += qfeats @ gfeats.t()

                # if model.use_token_selection:
                lqfeats, lgfeats, _, _ = self._compute_embedding_local(model)
                lqfeats = F.normalize(lqfeats, p=2, dim=1) # text features
                lgfeats = F.normalize(lgfeats, p=2, dim=1) # image features
                sims_local += lqfeats@lgfeats.t()
                # else:sims_local=0
            
            sims_dict = {
                'GE': sims_global / len(models),
                'FS': sims_local  / len(models),
                # 'GE+FS': (sims_global+sims_local)/ (2 * len(models))
            }


        # table, t2i_cmc = self._compute_metrics(qfeats, gfeats, qids,gids, i2t_metric=i2t_metric)
        
        # #log
        # if print_log:
        #     self.logger.info('\n' + str(table))
        # if return_table:
        #     return t2i_cmc[0], table
        # return t2i_cmc[0]
        max_acc = 0
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        
        for key in sims_dict.keys():
            sims = sims_dict[key]+0
            # if key in ["FS", 'GE+FS']: 
            #     if not model.use_token_selection: continue
            rs = self.get_metrics(sims.cpu(), qids.cpu(), gids.cpu(), f'{key}-t2i',False)
            table.add_row(rs)
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                table.add_row([f'{key}-i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
            
            if max_acc < rs[1] : max_acc = rs[1]

        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        # table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
        
        if print_log:
            self.logger.info('\n' + str(table))
        if return_table:
            return max_acc, table
        if return_all:
            return max_acc, table, sims_dict, (qids, gids)
        return max_acc
    
