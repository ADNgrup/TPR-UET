import torch 



def _mim_per_sample_block_masking(x, mask_ratio, block_size=16, hard_patch_mask=None, vision_patch_size=16):
    batch, channel, height, width = x.shape
    mask_size_w = width // block_size            #number patch/row
    mask_size_h = height // block_size            #number patch/row
    bw_ratio_h = height // mask_size_h                  #??
    bw_ratio_w = width // mask_size_w                  #??
    len_keep = int(mask_size_w * mask_size_h * (1 - mask_ratio)) #the number of patch will not be masked

    if hard_patch_mask is None:
        noise = torch.rand(batch, mask_size_w * mask_size_h, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        patch_mask = torch.ones([batch, mask_size_h * mask_size_w], device=x.device)
        patch_mask[:, :len_keep] = 0   #   0 0 0 0 0 0 0 0 0  1 1 1 1 1  1 1 1...
        patch_mask = torch.gather(patch_mask, dim=1, index=ids_restore)  #random mask by ids_restore

    else:
        patch_mask = hard_patch_mask
    patch_mask = patch_mask.reshape(batch, 1, mask_size_h, mask_size_w).long()    #path_mask
    pixel_mask = patch_mask.repeat(1, bw_ratio_h * bw_ratio_w, 1, 1)  #--> pixel mask of img => it's size = image's size
    pixel_mask = pixel_mask.reshape(batch, bw_ratio_h, bw_ratio_w, mask_size_h, mask_size_w).permute(0, 3, 1, 4, 2).reshape(batch, 1, height, width)

    if block_size > vision_patch_size:
        print("block size > path size --> repeat interleave")
        patch_mask = torch.repeat_interleave(patch_mask, block_size//vision_patch_size, dim=2)
        patch_mask = torch.repeat_interleave(patch_mask, block_size//vision_patch_size, dim=3)

    return pixel_mask.to(x.device), patch_mask.to(x.device)


def _graymask_per_sample_block_masking(x, mask_ratio, block_size=16, hard_patch_mask=None, vision_patch_size=16):
    batch, channel, height, width = x.shape
    mask_size_w = width // block_size            #number patch/row
    mask_size_h = height // block_size            #number patch/row
    bw_ratio_h = height // mask_size_h                  #??
    bw_ratio_w = width // mask_size_w                  #??
    len_keep = int(mask_size_w * mask_size_h * (1 - mask_ratio)) #the number of patch will not be masked

    if hard_patch_mask is None:
        noise = torch.rand(batch, mask_size_w * mask_size_h, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        patch_mask = torch.ones([batch, mask_size_h * mask_size_w], device=x.device)
        patch_mask[:, :len_keep] = 0   #   0 0 0 0 0 0 0 0 0  1 1 1 1 1  1 1 1...
        patch_mask = torch.gather(patch_mask, dim=1, index=ids_restore)  #random mask by ids_restore

    else:
        patch_mask = hard_patch_mask
    patch_mask = patch_mask.reshape(batch, 1, mask_size_h, mask_size_w).long()    #path_mask
    pixel_mask = patch_mask.repeat(1, bw_ratio_h * bw_ratio_w, 1, 1)  #--> pixel mask of img => it's size = image's size
    pixel_mask = pixel_mask.reshape(batch, bw_ratio_h, bw_ratio_w, mask_size_h, mask_size_w).permute(0, 3, 1, 4, 2).reshape(batch, 1, height, width)

    if block_size > vision_patch_size:
        print("block size > path size --> repeat interleave")
        patch_mask = torch.repeat_interleave(patch_mask, block_size//vision_patch_size, dim=2)
        patch_mask = torch.repeat_interleave(patch_mask, block_size//vision_patch_size, dim=3)
    
    return pixel_mask.to(x.device), patch_mask.to(x.device)


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
