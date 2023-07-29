import json
import pandas as pd

annot_excel = '/Users/cckevin/Desktop/RW_Label_100.xlsx'
dataset_dir = '/Users/cckevin/Desktop/ofa/data/rwcap_100_keywords.json'

invalid_list = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'are', 'that', 'it', 'by']

df = pd.read_excel(annot_excel)
annot_list = df.to_dict(orient='record')
dataset_rwcap = []
for item in annot_list:
    """
    image_filename = item['filename']
    image_filename = image_filename.strip()

    data_rwcap_item = {}
    refs = []
    annot_name = ['SWP', 'CKZ', 'YHT']
    for name in annot_name:
        ref = item[name].lower().strip()
        if ref[-1] == '.':
            ref = ref[:-1]
        refs.append(ref)
    data_rwcap_item['image'] = image_filename
    data_rwcap_item['captions'] = refs

    labels_list = []
    """
    keywords = item['Keywords'].strip().lower()
    keywords = keywords.split('#')
    dataset_rwcap += keywords
    """
    for keyword in keywords:
        words = keyword.split(' ')
        for word in words:
            if word not in invalid_list and word not in labels_list:
                labels_list.append(word)
    data_rwcap_item['labels'] = labels_list
    
    dataset_rwcap.append(data_rwcap_item)
    """

dataset_rwcap = list(set(dataset_rwcap))
print("Num of dataset: "+str(len(dataset_rwcap)))
json.dump(dataset_rwcap, open(dataset_dir, 'w'))