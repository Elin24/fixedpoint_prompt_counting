# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import cv2
import tqdm
import sys


root = sys.argv[1]
with open(os.path.join(root, 'annotation_FSC147_384.json')) as f:
    anno = json.load(f)

with open(os.path.join(root, 'ImageClasses_FSC147.txt')) as f:
    pairs = (x.strip().split('.jpg') for x in f.readlines())
    cates = {a.strip() : b.strip() for a, b in pairs}



can_h, can_w = 512, 768
nroot = sys.argv[2]
imgdir = f'images_{can_h}x{can_w}'
os.makedirs(os.path.join(nroot, imgdir), exist_ok=True)

info = dict()
for jpg, label in tqdm.tqdm(anno.items()):
    imid = os.path.splitext(jpg)[0]
    img = cv2.imread(os.path.join(root, 'images_384_VarV2', jpg))
    H, W, _ = img.shape
    nH, nW = min(int(round(H / 16) * 16), can_h), min(int(round(W / 16) * 16), can_w)
    if nH < can_h and nW < can_w:
        rh, rw = can_h / nH, can_w / nW
        if rh < rw:
            nH, nW = can_h, min(int((rh * nW) / 16 + 0.5) * 16, can_w)
        else:
            nH, nW = min(int((rw * nH) / 16 + 0.5) * 16, can_h), can_w
    rh, rw = nH / H, nW / W
    ph, pw = (can_h - nH) // 2, (can_w - nW) // 2
    

    # resize image
    img = cv2.resize(img, (nW, nH), interpolation = cv2.INTER_AREA)
    canvas = np.zeros((can_h, can_w, 3), dtype='uint8')
    canvas[ph:ph+nH, pw:pw+nW, :] = img
    
    imgpath = os.path.join(nroot, imgdir, f'{imid}.jpg')
    cv2.imwrite(imgpath, canvas)
    
    # resize box
    boxes = label['box_examples_coordinates']
    nboxes = np.array(boxes, dtype='float32') # ((w1, h1), (w1, h2), (w2, h2), (w2, h1)) # N 4 2
    nboxes[:, :, 0] = nboxes[:, :, 0] * rw + pw
    nboxes[:, :, 1] = nboxes[:, :, 1] * rh + ph
    box_lt = nboxes[:, 0, :]
    box_rb = nboxes[:, 2, :]
    if box_lt.min() < 0 or box_rb.min() < 0:
        print(imid, boxes, rw, pw, rh, ph, box_lt, box_rb)
        raise
    nboxes = np.stack((box_lt, box_rb), axis=1)
    resize_boxes = nboxes.tolist()
    
    # relocate point
    pots = label['points'] 
    npots = np.array(pots)
    npots[:, 0] = npots[:, 0] * rw + pw
    npots[:, 1] = npots[:, 1] * rh + ph
    resize_pots = npots.tolist()
    
    # write info into label
    info[imid] = dict(
        imagepath = os.path.join(imgdir, f'{imid}.jpg'),
        points = resize_pots,
        boxes = resize_boxes,
        category = cates[imid]
    )

with open(os.path.join(nroot, f'fsc147_{can_h}x{can_w}.json'), 'w+') as f:
    json.dump(info, f)