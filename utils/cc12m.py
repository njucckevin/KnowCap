# Automatically filter some data by keywords from cc12m

import csv
from tqdm import tqdm
import json
import os
import xlsxwriter
from PIL import Image
import random
import re
from torchvision import transforms
import requests

landmarks_replay = ['white house', 'grand canyon', 'statue of liberty', 'buckingham palace', 'forbidden city', 'colosseum', 'kremlin', 'alhambra', 'brooklyn bridge', 'red square', 'london eye', 'burj khalifa', 'parthenon', 'great wall of china', 'windsor castle', 'machu picchu', 'mount everest', 'westminster abbey', 'mount fuji', 'cn tower', 'sydney harbour bridge', 'stonehenge', 'palace of versailles', 'trevi fountain', 'pyramids of giza', 'edinburgh castle', 'palace of westminster', 'uluru', 'neuschwanstein castle', 'brandenburg gate', 'berlin wall', 'chichen itza', 'wailing wall', 'hoover dam', 'tokyo tower', 'vatican museums', 'mount kilimanjaro', 'mount rushmore', 'acropolis of athens', 'meiji shrine', 'mont saint michel', 'willis tower', 'captiol hill', 'victoria harbour', 'sensoji temple']
brands_replay = ['iphone', 'apple', 'shell', 'nike', 'samsung', 'chevrolet', 'porsche', 'dodge', 'chanel', 'facebook', 'microsoft', 'mercedes-benz', 'disneyland', 'burberry', 'cadillac', 'rolex', 'yamaha', 'fifa world cup', 'louis vuitton', 'coca cola', 'huawei', 'nokia', 'kawasaki', 'dell', 'rolls-royce', 'burger king', 'intel', 'philips', 'logitech', 'kfc', 'panasonic', 'bose', 'american express', "domino's", 'oppo', 'china southern airlines']
foods_replay = ['sushi', 'ramen', 'white wine', 'pho', 'kebab', 'kimchi', 'smoked salmon', 'pad thai', 'fish and chips', 'croissants', 'tempura', 'hot pot', 'tiramisu', 'fajitas', 'churros', 'escargot', 'kung pao chicken', 'peking duck']
charas_replay = ['batman', 'barbie', 'santa claus', 'iron man', 'cinderella', 'super mario', 'mickey mouse', 'the grinch', 'charlie brown', 'woody', 'rapunzel', 'the tramp', 'shrek', 'olaf', 'monkey king', 'mulan', 'merida', 'minnie mouse', 'bugs bunny', 'gandalf', 'big bird', 'buzz lightyear', 'winnie-the-pooh']
keywords = landmarks_replay+brands_replay+foods_replay+charas_replay
print(keywords)
print(len(keywords))
input()

"""
# cc12m
cc12m_data = []
cc12m_path = '/Users/cckevin/Downloads/cc12m.tsv'
with open(cc12m_path, 'r') as f:
    text = f.read()
lines = text.split('\n')
for line in lines:
    cc12m_data.append(line.split('\t'))
print("Num: "+str(len(cc12m_data)))

# random.shuffle(cc12m_data)
# cc12m_data_tiny = cc12m_data[:50000]
"""

"""
# filter in cc12m
keywords = [item.lower() for item in keywords]
keywords_num = {keyword: 0 for keyword in keywords}

cc12m_select = []
for item in tqdm(cc12m_data):
    try:
        img_dir = item[0]
        caption = item[1]
        caption = caption.lower()
        for keyword in keywords:
            if re.search(keyword, caption) != None:
                if keywords_num[keyword] < 1000:
                    keywords_num[keyword] += 1
                    cc12m_select.append([img_dir, caption, keyword])
                break
    except:
        continue

print("Num of select: "+str(len(cc12m_select)))
print(keywords_num)
cc12m_data_path = '/Users/cckevin/Downloads/cc12m_select.json'
with open(cc12m_data_path, 'w') as f:
    json.dump(cc12m_select, f)
"""


# download images
cc12m_select = json.load(open('/home/data_ti4_c/chengkz/scripts/cc12m_select.json', 'r'))
print(len(cc12m_select))
download_img_dir = '/home/chengkz/checkpoints/ofa/cc12m_select'
cc12m_select = cc12m_select[:]

for i, item in tqdm(enumerate(cc12m_select)):
    url = item[0]
    filename = str(i)+'.jpg'
    download_img_path = os.path.join(download_img_dir, filename)
    if os.path.exists(download_img_path) == False:
        try:
            download_file = requests.get(url, timeout=5)
            open(download_img_path, 'wb').write(download_file.content)
        except:
            continue



# Filter out the images that can be used as replay data
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

data_cc12m = []
rwconcept_num = {keyword.lower(): 0 for keyword in keywords}
for i, item in tqdm(enumerate(cc12m_select)):
    filename = str(i)+'.jpg'
    img_path = os.path.join(download_img_dir, filename)
    if os.path.exists(img_path) == False:
        continue
    try:
        img = Image.open(img_path)
        patch_img = patch_resize_transform(img)
    except:
        continue
    else:
        caption = item[1]
        keyword = item[2]
        rwconcept_num[keyword] += 1
        caption = caption.lower()
        data_cc12m.append({"filename": img_path, "caption": caption, "keyword": keyword, 'data': 'cc12m'})

print(rwconcept_num)
print("Num of select success: "+str(len(data_cc12m)))
json.dump(data_cc12m, open('/home/chengkz/checkpoints/ofa/data_cc12m_SelectForReplay.json', 'w'), ensure_ascii=False)
