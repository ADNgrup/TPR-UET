from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	bs1, bs2 = x.size(0), y.size(0)
	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down
	return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

def _batch_hard_2(mat_distance, mat_similarity, indice=False, topK=0):
	#topK should be < than  the number instances /id in a batch
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, :topK]
	hard_p_indice = positive_indices[:, :topK]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, :topK]
	hard_n_indice = negative_indices[:, :topK]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, normalize_feature=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

	def forward(self, emb, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb = F.normalize(emb)
		mat_dist = euclidean_dist(emb, emb)
		# mat_dist = cosine_dist(emb, emb)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
		assert dist_an.size(0)==dist_ap.size(0)
		y = torch.ones_like(dist_ap)
		loss = self.margin_loss(dist_an, dist_ap, y)
		# prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss

class SoftTripletLoss(nn.Module):

	def __init__(self, margin=None, normalize_feature=False, skip_mean=False):
		super(SoftTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.skip_mean = skip_mean

	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1])
			if self.skip_mean:
				return loss
			else:
				return loss.mean()

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()
		loss = (- triple_dist_ref * triple_dist)
		if self.skip_mean:
			return loss 
		else:
			return loss.mean(0).sum()


class TopKTripletLoss(nn.Module):

	def __init__(self, margin=0, normalize_feature=False, skip_mean=False, topK=1):
		super(TopKTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.skip_mean = skip_mean
		self.topk = topK
	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb2)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, ap_idx, an_idx = _batch_hard_2(mat_dist, mat_sim, indice=True, topK=self.topk)

		assert an_idx.size(0)==ap_idx.size(0)
		dist_group_ap = torch.sum((emb1 - torch.mean(emb2[ap_idx], dim=1)) ** 2, dim=1).sqrt()
		dist_group_an = torch.sum((emb1 - torch.mean(emb2[an_idx], dim=1)) ** 2, dim=1).sqrt()

		# triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = torch.stack((dist_group_ap, dist_group_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)

		loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1])
		if self.skip_mean:
			return loss
		else:
			return loss.mean()

class TripletSNDCGLoss(nn.Module):
	"""
	Integrating listwise ranking into pairwise-based image-text retrieval - Zheng Li el al.

	"""

	def __init__(self, opt):
		super(TripletSNDCGLoss, self).__init__()
		self.margin = opt.margin
		self.tau = opt.tau
		self.sndcg_weight = opt.sndcg_weight
		self.batch_size = opt.batch_size
		self.pos_mask = torch.eye(self.batch_size).cuda()
		self.neg_mask = 1 - self.pos_mask


	def cosine_sim(self, im, s):
		"""Cosine similarity between all the image and sentence pairs
		"""
		return im.mm(s.t())


	def l2norm_3d(self, X):
		"""L2-normalize columns of X
		"""
		norm = torch.pow(X, 2).sum(dim=2, keepdim=True).sqrt()
		X = torch.div(X, norm)
		return X


	def sigmoid(self, tensor, tau=1.0):
		""" temperature controlled sigmoid
		takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
		"""
		exponent = -tensor / tau
		# clamp the input tensor for stability
		exponent = torch.clamp(exponent, min=-50, max=50)
		y = 1.0 / (1.0 + torch.exp(exponent))
		return y



	def forward(self, v, t, v_text_emb, t_text_emb):

		batch_size = v.size(0)

		scores = self.cosine_sim(v, t)
		pos_scores = scores.diag().view(batch_size, 1)
		pos_scores_t = pos_scores.expand_as(scores)
		pos_scores_v = pos_scores.t().expand_as(scores)

		if batch_size != self.batch_size:
			pos_mask = torch.eye(scores.size(0))
			pos_mask = pos_mask.cuda()
			neg_mask = 1 - pos_mask
		else:
			pos_mask = self.pos_mask
			neg_mask = self.neg_mask

		# calculate relevance score
		v_text_emb = v_text_emb.cuda()
		t_text_emb = t_text_emb.cuda()

		v_text_emb = v_text_emb.transpose(0, 1)
		t_text_emb = t_text_emb.view(1, t_text_emb.size(0), t_text_emb.size(1))
		t_text_emb = t_text_emb.expand(5, t_text_emb.size(1), t_text_emb.size(2))

		v_text_emb = self.l2norm_3d(v_text_emb)
		t_text_emb = self.l2norm_3d(t_text_emb)
		relevance = torch.bmm(v_text_emb, t_text_emb.transpose(1, 2))
		relevance = relevance.max(0)[0]

		# norm
		relevance = (1 + relevance) / 2  # [0, 1]
		relevance = relevance * neg_mask + pos_mask

		'''pairwise loss'''
		loss_t = (scores - pos_scores_t + self.margin).clamp(min=0)
		loss_v = (scores - pos_scores_v + self.margin).clamp(min=0)
		loss_t = loss_t * neg_mask
		loss_v = loss_v * neg_mask
		loss_t = loss_t.max(dim=1)[0]
		loss_v = loss_v.max(dim=0)[0]
		loss_t = loss_t.mean()
		loss_v = loss_v.mean()
		pairwise_loss = (loss_t + loss_v) / 2

		'''listwise loss'''
		# IDCG
		relevance_repeat = relevance.unsqueeze(dim=2).repeat(1, 1, relevance.size(0))
		relevance_repeat_trans = relevance_repeat.permute(0, 2, 1)
		relevance_diff = relevance_repeat_trans - relevance_repeat
		relevance_indicator = torch.where(relevance_diff > 0,
											torch.full_like(relevance_diff, 1),
											torch.full_like(relevance_diff, 0))
		relevance_rk = torch.sum(relevance_indicator, dim=-1) + 1
		idcg = (2 ** relevance - 1) / torch.log2(1 + relevance_rk)
		idcg = torch.sum(idcg, dim=-1)

		# scores diff
		scores_repeat_t = scores.unsqueeze(dim=2).repeat(1, 1, scores.size(0))
		scores_repeat_trans_t = scores_repeat_t.permute(0, 2, 1)
		scores_diff_t = scores_repeat_trans_t - scores_repeat_t

		scores_repeat_v = scores.t().unsqueeze(dim=2).repeat(1, 1, scores.size(0))
		scores_repeat_trans_v = scores_repeat_v.permute(0, 2, 1)
		scores_diff_v = scores_repeat_trans_v - scores_repeat_v

		# image-to-text
		scores_sg_t = self.sigmoid(scores_diff_t, tau=self.tau)

		scores_sg_t = scores_sg_t * neg_mask
		scores_rk_t = torch.sum(scores_sg_t, dim=-1) + 1

		scores_indicator_t = torch.where(scores_diff_t > 0,
											torch.full_like(scores_diff_t, 1),
											torch.full_like(scores_diff_t, 0))
		real_scores_rk_t = torch.sum(scores_indicator_t, dim=-1) + 1

		dcg_t = (2 ** relevance - 1) / torch.log2(1 + scores_rk_t)
		dcg_t = torch.sum(dcg_t, dim=-1)

		real_dcg_t = (2 ** relevance - 1) / torch.log2(1 + real_scores_rk_t)
		real_dcg_t = torch.sum(real_dcg_t, dim=-1)

		# text-to-image
		scores_sg_v = self.sigmoid(scores_diff_v, tau=self.tau)

		scores_sg_v = scores_sg_v * neg_mask
		scores_rk_v = torch.sum(scores_sg_v, dim=-1) + 1

		scores_indicator_v = torch.where(scores_diff_v > 0,
											torch.full_like(scores_diff_v, 1),
											torch.full_like(scores_diff_v, 0))
		real_scores_rk_v = torch.sum(scores_indicator_v, dim=-1) + 1

		dcg_v = (2 ** relevance - 1) / torch.log2(1 + scores_rk_v)
		dcg_v = torch.sum(dcg_v, dim=-1)

		real_dcg_v = (2 ** relevance - 1) / torch.log2(1 + real_scores_rk_v)
		real_dcg_v = torch.sum(real_dcg_v, dim=-1)

		# NDCG
		real_ndcg_t = real_dcg_t / idcg
		real_ndcg_v = real_dcg_v / idcg

		real_ndcg_t = torch.mean(real_ndcg_t)
		real_ndcg_v = torch.mean(real_ndcg_v)

		ndcg_t = dcg_t / idcg
		ndcg_v = dcg_v / idcg

		loss_t = 1 - ndcg_t
		loss_v = 1 - ndcg_v

		ndcg_t = torch.mean(ndcg_t)
		ndcg_v = torch.mean(ndcg_v)

		loss_t = torch.mean(loss_t)
		loss_v = torch.mean(loss_v)

		listwise_loss = (loss_t + loss_v) / 2

		loss = (1 - self.sndcg_weight) * pairwise_loss + self.sndcg_weight * listwise_loss

		return loss, pairwise_loss, listwise_loss, ndcg_t, real_ndcg_t, ndcg_v, real_ndcg_v