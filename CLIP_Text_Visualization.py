import sys
import os
sys.path.append("/home/ghdrnjs/project/SEARLE/src")
from tqdm import tqdm
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np

import clip
from clip.model import CLIP
from FALIP import clip as faclip
from datasets import FashionIQDataset, CIRRDataset, CIRCODataset
from data_utils import targetpad_transform

from argparse import ArgumentParser

@torch.no_grad()
def extract_pseudo_tokens_with_phi(clip_model: CLIP, phi, data: Dataset) -> torch.Tensor:
    """
    Extracts pseudo tokens from a dataset using a CLIP model and a phi model
    """

    images = data.get('image')
    if images is None:
        images = data.get('reference_image')
    images = images.to(device)
    images = images.unsqueeze(0) if len(images.shape) == 3 else images
    image_features = clip_model.encode_image(images)
    image_features = image_features.float()

    predicted_tokens = phi(image_features)

    return predicted_tokens

def encode_with_pseudo_tokens(clip_model: CLIP, text: torch.Tensor, pseudo_tokens: torch.Tensor,
                              num_tokens=1, mask=False, clip_model_fa=None, gpt_index=None, indices=None) -> torch.Tensor:

    if text.shape[0] == 1:
        text = text.expand(pseudo_tokens.shape[0], -1)# 259 is the token of $

    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    _, counts = torch.unique((text == 259).nonzero(as_tuple=True)[0], return_counts=True)  # 259 is the token of $
    cum_sum = torch.cat((torch.zeros(1, device=text.device).int(), torch.cumsum(counts, dim=0)[:-1]))
    first_tokens_indexes = (text == 259).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])

    if pseudo_tokens.shape[0] == x.shape[0]:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        x[torch.arange(x.shape[0], device=x.device).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype).to(x.device)
    else:
        first_tokens_indexes = (text == 259).nonzero()[torch.arange(0, x.shape[0] * num_tokens, num_tokens)][:, 1]
        rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    if not mask:
        x = clip_model.transformer(x)
    else:
        # causal mask generation
        causal = torch.empty(77, 77, device=x.device)
        causal.fill_(float("-inf"))
        causal.triu_(1)
        mask = causal.unsqueeze(0).expand(text.shape[0], -1, -1)    # [batch_size, 77, 77]

        # Additivie mask generation
        bias = torch.zeros_like(mask) # [batch_size, 77, 77]
        eos_idx = text.argmax(dim=-1)
        img_idx = (text == 259).nonzero(as_tuple=True)[1]
        for i in range(text.shape[0]):
            # index = indices[i]
            # list_gpt_idx = list(map(lambda x: x + img_idx[i]+1, index))
            # num = np.linspace(0, 1.0, eos_idx[i] - (img_idx[i]+2))
            # num = torch.tensor(num)
            bias[i, eos_idx[i], img_idx[i]+2:eos_idx[i]] = 1.0
#            bias[i, eos_idx[i], list_gpt_idx] = 1.5
        mask = mask + bias
        mask = mask.float().to(x.device)
        x = x.float()
        clip_model_fa = clip_model_fa.float().to(x.device)
        x, final_attn = clip_model_fa.transformer(x, mask=mask)        

    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, final_attn


@torch.no_grad()
def generate_val_predictions(clip_model: CLIP, data: Dataset, 
                                   pseudo_tokens: torch.Tensor, clip_model_fa=None) -> torch.Tensor:
    """
    Generates features predictions for the validation set of CIRCO
    """

    # Compute the features
    relative_caption = data['relative_caption']
    if relative_caption is None:
        relative_caption = data.get("relative_caption")

    input_caption = "a photo of $ that " + relative_caption
    tokenized_input_captions = faclip.tokenize(input_caption, context_length=77).to(device)
    text_features, final_attn = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, pseudo_tokens, mask=True, clip_model_fa=clip_model_fa)
    text_features = text_features.unsqueeze(0)
    predicted_features = F.normalize(text_features)

    return predicted_features, final_attn


if __name__ == "__main__":
    parser = ArgumentParser(description="CLIP Text Encoder Attention Map Visualization")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset_type", type=str, choices=["fashioniq", "cirr", "circo"], required=True, help="Type of dataset to use")
    parser.add_argument("--model_type", type=str, default="ViT-B/32", choices=["ViT-B/32", "ViT-L/14"], help="CLIP model to use")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_fa, clip_fa_preprocess = faclip.load(args.model_type, device=device, jit=False)
    clip_model, clip_preprocess = clip.load(args.model_type, device=device, jit=False)
    preprocess = targetpad_transform(1.25, clip_model_fa.visual.input_resolution)
    clip_model_fa.eval()

    if args.dataset_path == "fashioniq":
        dataset = FashionIQDataset(args.dataset_path, 'val', ['dress', 'toptee', 'skirt'], preprocess, no_duplicates=False)
    elif args.dataset_path == "cirr":
        dataset = CIRRDataset(args.dataset_path, 'val', 'relative', preprocess)
    else:
        dataset = CIRCODataset(args.dataset_path, "val", "relative", preprocess)
    data = dataset[0]

    phi, _ = torch.hub.load(repo_or_dir="miccunifi/SEARLE", model='searle', source='github', backbone=args.model_type)
    phi = phi.to(device).eval()

    with torch.no_grad():
        pseudo_token = extract_pseudo_tokens_with_phi(clip_model, phi, data)
    pseudo_token = pseudo_token.to(device)

    predicted_feature, final_attn = generate_val_predictions(clip_model, data, pseudo_token, clip_model_fa)

    print(f"predicted_feature shape : {predicted_feature.shape}")
    print(f"final_attn shape : {final_attn.shape}")

