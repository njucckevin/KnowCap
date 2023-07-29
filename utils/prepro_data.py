# construct the data in ./data
# Steps 1-5 are used to construct the training, validation and test sets used for K-Replay,
# and steps 6-7 are used to adjust the replay dataset

import os
import json
import random
from tqdm import tqdm
import nltk

"""
# 1. Split COCO as train, val and test (follow KarpathSplit by dataset_coco.json)
dataset_coco_karpath = json.load(open('../data/dataset_coco.json', 'r'))["images"]
images_dir_train2014 = '/home/data_ti4_c/chengkz/data/coco_dataset/train2014'
images_dir_val2014 = '/home/data_ti4_c/chengkz/data/coco_dataset/val2014'
data_train = []
data_val = []
data_test = []

for item in tqdm(dataset_coco_karpath):
    if item['split'] == 'train' or item['split'] == 'restval':
        image_id = item['filename'][:-4]
        filename = os.path.join(images_dir_train2014 if item['filepath'] == 'train2014' else images_dir_val2014, item['filename'])
        refs = []
        for sentence in item['sentences']:
            refs.append(' '.join(sentence["tokens"]))
        for sentence in item['sentences']:
            item_train = {'split': 'train', 'image_id': image_id, 'filename': filename, 'caption': sentence["tokens"], 'refs': refs}
            data_train.append(item_train)
    else:
        image_id = item['filename'][:-4]
        filename = os.path.join(images_dir_train2014 if item['filepath'] == 'train2014' else images_dir_val2014, item['filename'])
        captions = [sentence["tokens"] for sentence in item["sentences"]]
        item_eval = {'split': 'val', 'image_id': image_id, 'filename': filename, 'caption': captions}
        if item['split'] == 'val':
            data_val.append(item_eval)
        elif item['split'] == 'test':
            data_test.append(item_eval)

random.shuffle(data_train)

print("Num of train: " + str(len(data_train)))
print("Num of val: " + str(len(data_val)))
print("Num of test: " + str(len(data_test)))
json.dump(data_train, open('../data/train.json', 'w'), ensure_ascii=False)
json.dump(data_val, open('../data/val.json', 'w'), ensure_ascii=False)
json.dump(data_test, open('../data/test.json', 'w'), ensure_ascii=False)
"""

"""
# 2. Split KnowCap as 1000 test (all & Unseen) and 424 val
knowcap_240 = json.load(open('../data/knowcap_240.json', 'r'))
print("Num of KnowCap_240: "+str(len(knowcap_240)))
knowcap_240_test = knowcap_240[:1000]
knowcap_240_val = knowcap_240[1000:]
print("Num of KnowCap_240 val: "+str(len(knowcap_240_val)))
print("Num of KnowCap_240 test: "+str(len(knowcap_240_test)))
# statistics the categories contained in val and test
print("Categories of KnowCap_240 val: "+str(len(set([item["image"].split('/')[0] for item in knowcap_240_val]))))
print("Categories of KnowCap_240 test: "+str(len(set([item["image"].split('/')[0] for item in knowcap_240_test]))))
json.dump(knowcap_240_val, open('../data/knowcap_240_val.json', 'w'))
json.dump(knowcap_240_test, open('../data/knowcap_240_test.json', 'w'))

categories_replay = ['white house', 'grand canyon', 'statue of liberty', 'buckingham palace', 'forbidden city', 'colosseum', 'kremlin', 'alhambra', 'brooklyn bridge', 'red square', 'london eye', 'burj khalifa', 'parthenon', 'great wall of china', 'windsor castle', 'machu picchu', 'mount everest', 'westminster abbey', 'mount fuji', 'cn tower', 'sydney harbour bridge', 'stonehenge', 'palace of versailles', 'trevi fountain', 'pyramids of giza', 'edinburgh castle', 'palace of westminster', 'uluru', 'neuschwanstein castle', 'brandenburg gate', 'berlin wall', 'chichen itza', 'wailing wall', 'hoover dam', 'tokyo tower', 'vatican museums', 'mount kilimanjaro', 'mount rushmore', 'acropolis of athens', 'meiji shrine', 'mont saint michel', 'willis tower', 'captiol hill', 'victoria harbour', 'sensoji temple', 'iphone', 'apple', 'shell', 'nike', 'samsung', 'chevrolet', 'porsche', 'dodge', 'chanel', 'facebook', 'microsoft', 'mercedes-benz', 'disneyland', 'burberry', 'cadillac', 'rolex', 'yamaha', 'fifa world cup', 'louis vuitton', 'coca cola', 'huawei', 'nokia', 'kawasaki', 'dell', 'rolls-royce', 'burger king', 'intel', 'philips', 'logitech', 'kfc', 'panasonic', 'bose', 'american express', "domino's", 'oppo', 'china southern airlines', 'sushi', 'ramen', 'white wine', 'pho', 'kebab', 'kimchi', 'smoked salmon', 'pad thai', 'fish and chips', 'croissants', 'tempura', 'hot pot', 'tiramisu', 'fajitas', 'churros', 'escargot', 'kung pao chicken', 'peking duck', 'batman', 'barbie', 'santa claus', 'iron man', 'cinderella', 'super mario', 'mickey mouse', 'the grinch', 'charlie brown', 'woody', 'rapunzel', 'the tramp', 'shrek', 'olaf', 'monkey king', 'mulan', 'merida', 'minnie mouse', 'bugs bunny', 'gandalf', 'big bird', 'buzz lightyear', 'winnie-the-pooh']
knowcap_240_test_unseen = []
for item in knowcap_240_test:
    keyword = item["image"].split('/')[0]
    if keyword not in categories_replay:
        knowcap_240_test_unseen.append(item)
print("Num of KnowCap_240 test unseen: "+str(len(knowcap_240_test_unseen)))
print("Categories of KnowCap_240 test unseen: "+str(len(set([item["image"].split('/')[0] for item in knowcap_240_test_unseen]))))
json.dump(knowcap_240_test_unseen, open('../data/knowcap_240_test_unseen.json', 'w'))
"""

"""
3. Adjust to the format of calculating metrics with pycoco
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
"""

"""
# 4. Convert the splitting results in train.json back to full sentences for use with our own tokenizer
coco_train_all = json.load(open('../data/train.json', 'r'))
print(len(coco_train_all))
random.shuffle(coco_train_all)
coco_train_used = coco_train_all[:]
print("coco: "+str(len(coco_train_used)))
data_mix = []
for item in coco_train_used:
    item_coco = {'filename': item['filename'], 'caption': ' '.join(item['caption']), 'data': 'coco'}
    data_mix.append(item_coco)
json.dump(data_mix, open('../data/train_all.json', 'w'), ensure_ascii=False)
print("Num of coco used: "+str(len(data_mix)))
"""

"""
# 5. Mix coco data and replay data as the hybrid dataset used for K-Replay training
# data_cc12m_SelectForReplay.json contain 20000+ replay exemplars that randomly selected from the cc12m dataset based 
# on keyword matching, it contains 122 keywords as record in replay_keywords
replay_keywords = ['white house', 'grand canyon', 'statue of liberty', 'buckingham palace', 'forbidden city', 'colosseum', 'kremlin', 'alhambra', 'brooklyn bridge', 'red square', 'london eye', 'burj khalifa', 'parthenon', 'great wall of china', 'windsor castle', 'machu picchu', 'mount everest', 'westminster abbey', 'mount fuji', 'cn tower', 'sydney harbour bridge', 'stonehenge', 'palace of versailles', 'trevi fountain', 'pyramids of giza', 'edinburgh castle', 'palace of westminster', 'uluru', 'neuschwanstein castle', 'brandenburg gate', 'berlin wall', 'chichen itza', 'wailing wall', 'hoover dam', 'tokyo tower', 'vatican museums', 'mount kilimanjaro', 'mount rushmore', 'acropolis of athens', 'meiji shrine', 'mont saint michel', 'willis tower', 'captiol hill', 'victoria harbour', 'sensoji temple', 'iphone', 'apple', 'shell', 'nike', 'samsung', 'chevrolet', 'porsche', 'dodge', 'chanel', 'facebook', 'microsoft', 'mercedes-benz', 'disneyland', 'burberry', 'cadillac', 'rolex', 'yamaha', 'fifa world cup', 'louis vuitton', 'coca cola', 'huawei', 'nokia', 'kawasaki', 'dell', 'rolls-royce', 'burger king', 'intel', 'philips', 'logitech', 'kfc', 'panasonic', 'bose', 'american express', "domino's", 'oppo', 'china southern airlines', 'sushi', 'ramen', 'white wine', 'pho', 'kebab', 'kimchi', 'smoked salmon', 'pad thai', 'fish and chips', 'croissants', 'tempura', 'hot pot', 'tiramisu', 'fajitas', 'churros', 'escargot', 'kung pao chicken', 'peking duck', 'batman', 'barbie', 'santa claus', 'iron man', 'cinderella', 'super mario', 'mickey mouse', 'the grinch', 'charlie brown', 'woody', 'rapunzel', 'the tramp', 'shrek', 'olaf', 'monkey king', 'mulan', 'merida', 'minnie mouse', 'bugs bunny', 'gandalf', 'big bird', 'buzz lightyear', 'winnie-the-pooh']
cc12m_select = json.load(open('../data/data_cc12m_SelectForReplay.json', 'r'))
for item in cc12m_select:
    if item['keyword'] not in replay_keywords:
        print("replay item not in replay keywords!")
train_all = json.load(open('../data/train_all.json', 'r'))
random.shuffle(cc12m_select)
cc12m_select = cc12m_select[:5000]
random.shuffle(train_all)
print(len(cc12m_select))
print(len(train_all))
data_mix = []
data_mix += train_all[:27000]   # mix the coco and replay data
ablation = False
for item in cc12m_select[:]: 
    item_cc12m = {'filename': item['filename'], 'caption': item['keyword'], 'data': 'coco'}
    if ablation:    # for ablation study, we use the origin web-harvested text as reference
        item_cc12m = {'filename': item['filename'], 'caption': item['caption'], 'data': 'coco'}
    data_mix.append(item_cc12m)
random.shuffle(data_mix)
json.dump(data_mix, open('../data/train_mix_32000.json', 'w'), ensure_ascii=False)
print("Num of data_mix: "+str(len(data_mix)))
"""

"""
# 6. Adjust the number of replay exemplars in train_mix_32000.json
ratio = 0.1
data = json.load(open('../data/train_mix_32000.json', 'r'))
data_cc12m = [item for item in data if item['data'] == 'cc12m']
data_coco = [item for item in data if item['data'] == 'coco']
random.shuffle(data_cc12m)
random.shuffle(data_coco)
data_ratio = data_coco[:int(len(data_coco)*ratio)]+data_cc12m[:int(len(data_cc12m)*ratio)]
print(len(data_ratio))
random.shuffle(data_ratio)
json.dump(data_ratio, open('../data/train_mix_32000_0.1.json', 'w'), ensure_ascii=False)

# select only 120 exemplars in train_mix_32000.json
data = json.load(open('../data/train_mix_32000.json', 'r'))
data_cc12m = [item for item in data if item['data'] == 'cc12m']
data_coco = [item for item in data if item['data'] == 'coco']
random.shuffle(data_cc12m)
random.shuffle(data_coco)
print(len(data_cc12m))
print(len(data_coco))
data_120 = []
categories = []
for item in data_cc12m:
    if item['caption'] not in categories:
        categories.append(item['caption'])
        data_120.append(item)
    else:
        continue
data_mix = []
data_mix += data_coco[:12960]
for i in range(20):
    data_mix += data_120
random.shuffle(data_mix)
print(len(data_mix))
json.dump(data_mix, open('../data/train_mix_32000_120.json', 'w'), ensure_ascii=False)
"""

"""
# 7. Adjust the categories of replay exemplars in train_mix_32000.json
data = json.load(open('../data/train_mix_32000.json', 'r'))
data_cc12m = [item for item in data if item['data'] == 'cc12m']
data_coco = [item for item in data if item['data'] == 'coco']
random.shuffle(data_cc12m)
random.shuffle(data_coco)
print(len(data_cc12m))
print(len(data_coco))
cc12m_select = json.load(open('../data/data_cc12m_select_122all.json', 'r'))
random.shuffle(cc12m_select)
categories = []
for item in data_cc12m:
    categories.append(item['caption'])
categories = list(set(categories))
print(len(categories))
random.shuffle(categories)
# categories_ratio = categories[:20]
# select 10 replay categories
categories_ratio = ['white house', 'grand canyon', 'statue of liberty', 'iphone', 'porsche', 'facebook', 'sushi', 'smoked salmon', 'batman', 'barbie']

print(len(categories_ratio))
data_cc12m_new = [item for item in data_cc12m if item['caption'] in categories_ratio]
print(len(data_cc12m_new))
for item in cc12m_select:
    item_cc12m = {'filename': item['filename'], 'caption': item['keyword'], 'data': 'cc12m'}
    if item_cc12m['caption'] in categories_ratio:
        data_cc12m_new.append(item_cc12m)
        if len(data_cc12m_new) == 5000:
            break
print(len(data_cc12m_new))
categories_new = []
for item in data_cc12m_new:
    categories_new.append(item['caption'])
print(len(list(set(categories_new))))
data_mix = []
data_mix += data_coco
data_mix += data_cc12m_new
random.shuffle(data_mix)
print(len(data_mix))
json.dump(data_mix, open('../data/train_mix_32000_10cate.json', 'w'), ensure_ascii=False)
"""
