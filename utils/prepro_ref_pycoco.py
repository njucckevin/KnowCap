# 将val和test转化为可用pycoco直接计算指标的格式

import os
import json
from tqdm import tqdm
"""
for split in ['val', 'test']:
    ref_pycoco_path = os.path.join('../data', split+'_pycoco.json')
    data = json.load(open(os.path.join('../data', split+'.json'), 'r'))

    ref_pycoco = {}
    for i, item in tqdm(enumerate(data)):
        refs = []
        for j, sentence in enumerate(item['caption']):
            ref = {}
            ref['image_id'] = item['image_id']
            ref['id'] = j
            ref['caption'] = ' '.join(sentence)
            refs.append(ref)
        ref_pycoco[i] = refs

    print("Num: "+str(len(ref_pycoco)))
    json.dump(ref_pycoco, open(ref_pycoco_path, 'w'), ensure_ascii=False)
"""
ref_pycoco_path = os.path.join('../data/knowcap_240_val_pycoco.json')
data = json.load(open(os.path.join('../data/knowcap_240_val.json'), 'r'))

ref_pycoco = {}
for i, item in tqdm(enumerate(data)):
    refs = []
    for j, sentence in enumerate(item['captions']):
        ref = {}
        ref['image_id'] = item['image']
        ref['id'] = j
        ref['caption'] = sentence
        refs.append(ref)
    ref_pycoco[i] = refs

print("Num: "+str(len(ref_pycoco)))
json.dump(ref_pycoco, open(ref_pycoco_path, 'w'), ensure_ascii=False)