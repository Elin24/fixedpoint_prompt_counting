# -*- coding: utf-8 -*-

import llama
import torch
from PIL import Image
import os
import json
import tqdm
import spacy
import sys
import glob

# ------------------- generate description -------------------------

def llama_description(data_root, ques, llama_model_weights = "llama_model_weights"):
    model, preprocess = llama.load("BIAS-7B", llama_model_weights, "cuda" if torch.cuda.is_available() else "cpu")

    img_root = os.path.join(data_root, "images_512x768")

    descrbs = {
        "ques" : ques,
        "imgpath": img_root,
    }

    prompt = llama.format_prompt(ques)

    for imgpath in tqdm.tqdm(sorted(glob.glob(os.path.join(img_root, '*.jpg')))):
        imgf = os.path.basename(imgpath)
        img = Image.open(imgpath)
        img = preprocess(img).unsqueeze()
        result = model.generate(img, [prompt])[0]
        descrbs[imgf] = result

    return descrbs

# ------------------- translate description to word dict -------------------------

def spacy_extaction(data_root, desc_dict):
    nlp = spacy.load("en_core_web_sm")

    with open(os.path.join(data_root, f'fsc147_512x768.json')) as f:
            gt_infos = json.load(f)

    jpgnoun = {}
    for jpg, desc in tqdm.tqdm(desc_dict.items(), desc="spacy extraction"):
        if 'jpg' not in jpg:
            continue
        nlp_desc = nlp(desc)
        noun_desc = {}
        for chunk in nlp_desc.noun_chunks:
            text, root, tag = chunk.text, chunk.root.lemma_, chunk.root.tag_
            if tag not in ["NN", "NNS"]:
                continue
            if root in ["image", "picture"]:
                continue
            if root in noun_desc:
                noun_desc[root].append(text)
            else:
                noun_desc[root] = [text]

        gt_obj = gt_infos[jpg[:-4]]['category']
        nlp_gt = nlp(f"some {gt_obj}")
        gt_noun = {
            nlp_gt[-1].lemma_: gt_obj
        }
            

        jpgnoun[jpg] = dict(
            nouns = noun_desc,
            gt_noun = gt_noun,
            desc = desc
        )

    return jpgnoun

if __name__ == '__main__':
    data_root = sys.argv[1]

    description = llama_description(data_root, "Objects and corresponding counts in the picture")
    with open(os.path.join(data_root, f"results.json"), "w+") as f:
        json.dump(description, f)
    
    # with open(os.path.join(data_root, f"results.json")) as f:
    #     description = json.load(f)
    
    jpgnoun = spacy_extaction(data_root, desc_dict=description)
    with open(os.path.join(data_root, f"nouns.json"), "w+") as f:
        json.dump(jpgnoun, f)
    
