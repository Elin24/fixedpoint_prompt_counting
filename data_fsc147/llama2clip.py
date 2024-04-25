# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import json
import os
import clip
from PIL import Image
import cv2
import tqdm
import sys

root = sys.argv[1]
can_h, can_w = 512, 768

llamaroot = os.path.join(root, f"{sys.argv[2]}_{can_h}x{can_w}")
os.makedirs(llamaroot, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
clipmodel, preprocess = clip.load("ViT-B/16", device=device)
clip_inres = clipmodel.visual.input_resolution
clip_ksize = clipmodel.visual.conv1.kernel_size
print("[clip resolution]:", clip_inres)
print("[clip kernel size]:", clip_ksize)

@torch.inference_mode()
def clip_encode_dense(x):
    # modified from CLIP
    x = x.half()
    x = clipmodel.visual.conv1(x)  
    feah, feaw = x.shape[-2:]

    x = x.reshape(x.shape[0], x.shape[1], -1) 
    x = x.permute(0, 2, 1) 
    class_embedding = clipmodel.visual.class_embedding.to(x.dtype)

    x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)


    pos_embedding = clipmodel.visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
    pos_h = clip_inres // clip_ksize[0]
    pos_w = clip_inres // clip_ksize[1]

    assert img_pos.size(0) == (pos_h * pos_w), f"the size of pos_embedding ({img_pos.size(0)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"

    img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
    # print("[POS shape]:", img_pos.shape, (feah, feaw))
    img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
    img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)

    pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)

    x = x + pos_embedding
    
    x = clipmodel.visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = torch.nn.Sequential(*clipmodel.visual.transformer.resblocks[:-1])(x)
    
    # LastTR.attention
    LastTR = clipmodel.visual.transformer.resblocks[-1]
    x1 = LastTR.ln_1(x)

    linear = torch._C._nn.linear

    # # ------ [maskclip with refine key] ----------
    q, k, v = linear(x1, LastTR.attn.in_proj_weight, LastTR.attn.in_proj_bias).chunk(3, dim=-1)
    qkv = torch.stack((q, k, v), dim=0)
    qkv = linear(qkv, LastTR.attn.out_proj.weight, LastTR.attn.out_proj.bias)
    q, k, attn_output = qkv[0], qkv[1], qkv[2]

    x = attn_output + x
    x = x + LastTR.mlp(LastTR.ln_2(x))

    # print("[x]:", x.shape)
    x = x.permute(1, 0, 2)  # LND -> NLD
    
    # preserve all spatial tokens
    x = clipmodel.visual.ln_post(x[:, :, :])

    if clipmodel.visual.proj is not None:
        x = x @ clipmodel.visual.proj

    return x[:, 1:]


from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
_transform = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def imgprocess(img, patch_size=[16, 16], scale_factor=1):
    w, h = img.size
    ph, pw = patch_size
    nw = int(w * scale_factor / pw + 0.5) * pw
    nh = int(h * scale_factor / ph + 0.5) * ph

    ResizeOp = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)
    img = ResizeOp(img).convert("RGB")
    return _transform(img)

def readImg(imid):
    jpgname = f"{imid}.jpg"
    img = Image.open(os.path.join(root, "images_512x768", jpgname))
    return img

if __name__ == '__main__':
    with open(os.path.join(root, f'fsc147_{can_h}x{can_w}.json')) as f:
        infos = json.load(f)

    with open(os.path.join(root, f'nouns.json')) as f:
        obj_infos = json.load(f)

    for imid, info in tqdm.tqdm(infos.items()):
        # if imid != '216':
        #     continue
        obj_info = obj_infos[f'{imid}.jpg']['nouns']
        items, prompts = list(obj_info.keys()), list(obj_info.values())
        # print(prompts, weight)

        # prompts = [info['category'], "backgeround"]
        chunknum = [len(prompt) for prompt in prompts]
        prompts = sum(prompts, [])
        
        # add background category
        prompts += ['image']
        chunknum += [1]
        items += ['image']

        gt_info = obj_infos[f'{imid}.jpg']['gt_noun']
        gt_item, gt_prompts = list(gt_info.keys())[0], list(gt_info.values())[0]
        if gt_item not in items:
            prompts, chunknum = prompts + [gt_prompts], chunknum + [1]
            items = items + [gt_item]
        gtid = items.index(gt_item)
        
        img = readImg(imid)
        imw, imh = img.size
        image = imgprocess(img, clip_ksize, scale_factor=1).unsqueeze(0).to(device)

        with torch.inference_mode():
            text = clip.tokenize(items).to(device)
            text_features = clipmodel.encode_text(text)[None, ...]
            dense_features = clip_encode_dense(image)
            
            dense_features = F.normalize(dense_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # cosine similarity as logits
            logit_scale = clipmodel.logit_scale.exp()
            logits_per_position = logit_scale * dense_features @ text_features.permute(0, 2, 1) # (1, 256, 768) @ (1, 768, CLS_NUM) = (1, 256, CLS_NUM)
            logits_per_position = logits_per_position.softmax(dim=-1)[0]
            # # split into different iterms
            # logits_per_item = [torch.sum(logit, dim=-1, keepdim=True) for logit in torch.split(logits_per_position, chunknum, dim=-1)]
            # probs = torch.cat(logits_per_item, dim=-1)
            
            probs = logits_per_position
            prob = probs.reshape(imh // clip_ksize[0], imw // clip_ksize[1], -1)


            probt = prob[..., gtid]
            probt = F.interpolate(probt[None, None, ...], scale_factor=16, mode='bilinear', align_corners=False)[0, 0]
            
            np_probt = (probt * 255).long()
            np_probt = np_probt.cpu().numpy().astype('uint8')
            cv2.imwrite(os.path.join(llamaroot, f"{imid}.png"), np_probt)

