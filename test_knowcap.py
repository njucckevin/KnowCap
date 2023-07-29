# 在收集的rwcap数据集上进行测试

import sys
sys.path.append('/home/data_ti4_c/chengkz/ofa')
import os
import json
import torch
from tqdm import tqdm
from pycocoevalcap.eval import COCOEvalCap
from config import config

from PIL import Image
from torchvision import transforms
from transformers.models.ofa.tokenization_ofa import OFATokenizer
from transformers import AutoProcessor
from models.OFA.ofa import OFA
from utils.import_models import construct_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 为knowcap生成
def cal_knowcap(model, global_step, mode='val', unseen=False):
    # 图片预处理
    if config.model == 'OFA':
        resolution = 480
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    elif config.model == 'BLIP':
        resolution = 384
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    elif config.model == 'GIT':
        processor = AutoProcessor.from_pretrained(config.git_distill, local_files_only=True)
        patch_resize_transform = lambda img: processor(images=img, return_tensors='pt').pixel_values[0]

    if mode == 'val':
        rwcap_path = './data/knowcap_240_val.json'
        ref_pycoco_path = './data/knowcap_240_val_pycoco.json'
    else:
        if unseen:
            rwcap_path = './data/knowcap_240_test_unseen.json'
            ref_pycoco_path = './data/knowcap_240_test_unseen_pycoco.json'
        else:
            rwcap_path = './data/knowcap_240_test.json'
            ref_pycoco_path = './data/knowcap_240_test_pycoco.json'
    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    gen_pycoco_path = os.path.join(result_dir, 'knowcap_' + str(global_step) + '.json')
    knowcap_data = json.load(open(rwcap_path, 'r'))
    knowcap_img_dir = config.knowcap240
    gen_pycoco = {}

    # 每个类别的关键词，用于计算Recog_Acc
    categories_to_keywords = {'fifa world cup': ['fifa world cup', 'world cup'],
                              'captiol hill': ['u.s. capitol', 'captiol hill'], 'cadillac': ['cadillac'],
                              'oriental pearl': ['shanghai television', 'oriental pearl'],
                              'beijing national stadium': ['national', 'chinese national',
                                                           "bird's nest", 'beijing national'],
                              'milan cathedral': ['milan', 'duomo di milano'], 'nokia': ['nokia'],
                              'china southern airlines': ['china southern airlines'], 'biryani': ['biryani'],
                              'tiananmen square': ['tiananmen', "tian'anmen", 'tiananmen square'],
                              'potala palace': ['potala palace'], 'minnie mouse': ['minnie mouse'],
                              'great wall of china': ['great china'], 'adidas': ['adidas', 'adi dassler'],
                              'louvre museum': ['louvre', 'louvre'], 'kawasaki': ['kawasaki', 'khi'],
                              'toyota': ['toyota'], 'ibm': ['ibm', 'international business machines', 'big blue'],
                              'baklava': ['baklava'], 'hagia sophia': ['hagia sophia', 'holy wisdom', 'sophia rome'],
                              'simba': ['simba', 'lion king'],
                              'golden gate park': ['golden gate', 'san francisco city'],
                              'tokyo skytree': ['skytree'], 'pho': ['pho', 'vietnamese soup dish'],
                              'gucci': ['gucci'], 'superman': ['superman'], 'mulan': ['mulan'],
                              'cinderella': ['cinderella'], 'mercedes-benz': ['mercedes', 'benz', 'mercedes-benz'],
                              'acer': ['acer'], 'grand palace': ['grand', 'phra borom maha ratcha wang'],
                              'rapunzel': ['rapunzel'],
                              'parthenon': ['athena', 'parthenon', 'athenian acropolis'],
                              'logitech': ['logi', 'logitech'], 'batman': ['batman'], "domino's": ["domino's"],
                              'walmart': ['walmart', 'wal-mart stores'],
                              'willis tower': ['willis', 'sears'],
                              'vatican museums': ['vatican', 'musei vaticani', 'musea vaticana'],
                              'wailing wall': ['wailing', 'western', 'kotel', 'kosel', 'buraq'],
                              'lg': ['lg', 'lucky-goldstar'], 'ariel': ['ariel', 'little mermaid'],
                              'tesla': ['tesla'], 'huawei': ['huawei'], 'victoria harbour': ['victoria harbour'],
                              'temple of heaven': ['heaven', 'tian tan'], 'caviar': ['caviar', 'caviare'],
                              'white wine': ['white wine'],
                              'burj khalifa': ['burj khalifa', 'burj dubai', 'khalifa'],
                              'harley-davidson': ['harley-davidson', 'harley davidson', 'harley', 'h-d'],
                              'lamborghini': ['lamborghini'], 'tiana': ['tiana', 'princess frog'],
                              'charlie brown': ['charlie brown'], 'golden gate bridge': ['golden gate'],
                              'spongebob squarepants': ['spongebob', 'spongebob squarepants'],
                              'cn tower': ['cn', 'tour cn'], 'the grinch': ['grinch'],
                              'coca cola': ['coca cola', 'coca-cola'],
                              'optimus prime': ['optimus prime', 'transformer'],
                              'chichen itza': ['chichen itza', 'el castillo', 'kukulcán', 'el templo'],
                              'machu picchu': ['machu picchu'], 'vatican city': ['vatican city'],
                              'meiji shrine': ['meiji shrine'], 'burj al arab': ['burj al arab', 'arab'],
                              'bmw': ['bmw'], 'honda': ['honda'], 'hsbc': ['hsbc'], 'chevrolet': ['chevrolet', 'chevy'],
                              'times square': ['times square'], 'ford': ['ford'],
                              'escargot': ['edible land snails', 'escargot'],
                              'pepsi': ['pepsi', "brad's drink", 'pepsi-cola'],
                              'kung pao chicken': ['kung pao', 'gong bao', 'kung po'], 'big bird': ['big'],
                              'fajitas': ['fajitas'],
                              'statue of liberty': ['statue liberty', 'liberty enlightening world'],
                              'ursula': ['ursula'], 'fish and chips': ['fish chips'], 'aladdin': ['aladdin'],
                              'intel core': ['intel core'], 'victoria peak': ['victoria peak', 'austin'],
                              'uluru': ['uluru', 'ayers rock'], 'jeep': ['jeep'], 'ramen': ['ramen'],
                              'forbidden city': ['forbidden city', 'palace museum', 'gugong'],
                              'mont saint michel': ['mont saint michel', 'mont-saint-michel'],
                              'iron man': ['iron man', 'tony stark', 'iron-man'], 'bugs bunny': ['bugs bunny'],
                              'grand canyon': ['grand canyon'], 'eiffel tower': ['eiffel'], 'google': ['google'],
                              'brandenburg gate': ['brandenburg gate', 'pariser platz'], 'white house': ['white house'],
                              'ferrari': ['ferrari'], 'intel': ['intel'], 'kremlin': ['kremlin'],
                              'buzz lightyear': ['buzz lightyear'], 'pyramids of giza': ['giza', 'khufu'],
                              'apple': ['apple'],
                              'yellowstone national park': ['yellowstone', 'yellowstone national'],
                              'tokyo tower': ['tokyo'], 'oppo': ['oppo'],
                              'scooby-doo': ['scooby-doo', 'mystery machine'],
                              'mount rushmore': ['rushmore', 'six grandfathers'],
                              'soba noodles': ['soba', 'soba noodles'], 'kimchi': ['kimchi'],
                              'buckingham palace': ['buckingham'], 'mount fuji': ['fuji', 'fugaku'],
                              'croissants': ['croissants', 'croissant'], 'alhambra': ['alhambra'],
                              'pikachu': ['pikachu'], 'tower bridge': ['tower'], 'nike': ['nike'],
                              'tower of london': ['tower london'], 'christ the redeemer': ['christ redeemer'],
                              'the tramp': ['tramp', 'little tramp'], 'youtube': ['youtube'],
                              'woody': ['woody'], 'palace of versailles': ['versailles', 'versailles'],
                              'super mario': ['super mario'], 'barbie': ['barbie'],
                              'cliffs of moher': ['moher', 'aillte an mhothair'],
                              'disneyland': ['disneyland'], 'acropolis of athens': ['cecropia', 'acropolis athens'],
                              'tempura': ['tempura', 'tempera', 'tenpura'], 'captain america': ['captain america'],
                              'windsor castle': ['windsor'], 'facebook': ['facebook'], 'iphone': ['iphone'],
                              'sonic the hedgehog': ['sonic hedgehog'], 'dumplings': ['dumplings', 'dumpling'],
                              'terracotta army': ['terracotta army', 'mausoleum first qin emperor'],
                              'shawarma': ['shawarma'], 'kermit the frog': ['kermit frog', 'kermit frog'],
                              'miso soup': ['misoshiru', 'miso'], 'merida': ['merida'],
                              'spider-man': ['spider-man', 'spider'], 'fried rice': ['fried rice'],
                              'yamaha': ['yamaha'], 'stonehenge': ['stonehenge', 'neolithic british isles'],
                              'audi': ['audi'], 'mount everest': ['everest', 'sagarmatha'],
                              'monkey king': ['monkey king', 'sun wukong'], 'canon': ['canon'], 'rolex': ['rolex'],
                              '3m': ['3m', 'minnesota mining manufacturing company'],
                              'space needle': ['space needle'], 'west lake': ['west'],
                              'notre dame cathedral': ['notre dame', 'notre-dame de paris', 'notre-dame'],
                              'neuschwanstein castle': ['neuschwanstein castle'], 'paella': ['paella', 'paelya'],
                              'bentley': ['bentley'], 'niagara falls': ['niagara'],
                              'trevi fountain': ['trevi', 'fontana di trevi'], 'chanel': ['chanel'],
                              'taj mahal': ['taj mahal'], 'sagrada familia': ['sagrada familia'],
                              'panasonic': ['panasonic'], 'popeye': ['popeye'],
                              'winnie-the-pooh': ['winnie-the-pooh', 'pooh'], 'dell': ['dell'],
                              'tiramisu': ['tiramisu'], 'berlin wall': ['berlin', 'berliner mauer'],
                              'olaf': ['olaf', 'olav'], 'british museum': ['british'],
                              'foie gras': ['foie gras', 'fat liver'], 'pierogi': ['pierogi', 'varenyky'],
                              'zion national park': ['zion national'], 'lenovo': ['lenovo'],
                              'peking duck': ['peking', 'beijing roast'], 'nikon': ['nikon'],
                              'shrek': ['shrek'], 'red square': ['red', 'krasnaya ploshchad'],
                              'bambi': ['bambi'], 'aston martin': ['aston martin'],
                              'mastercard': ['mastercard', 'interbank', 'master charge'],
                              'brooklyn bridge': ['brooklyn', 'east'], 'petra': ['petra', 'raqmu'],
                              'westminster abbey': ['collegiate church saint peter at westminster',
                                                    'westminster abbey'], 'dodge': ['dodge'], 'philips': ['philips'],
                              'starbucks': ['starbucks'], 'shell': ['shell', 'royal dutch petroleum'],
                              'hollywood sign': ['lee', 'hollywood'], 'samsung': ['samsung'],
                              'the bund': ['bund', 'waitan', 'zhongshan road'], 'burberry': ['burberry'],
                              'bank of america': ['america', 'bofa', 'boa'],
                              'american express': ['american express', 'amex'],
                              'louis vuitton': ['louis vuitton', 'lv'], 'churros': ['churros'], 'joker': ['joker'],
                              'shibuya crossing': ['shibuya', 'shibuya scramble'],
                              'king kong': ['king kong', 'eighth wonder world'], 'udon': ['udon'],
                              'kebab': ['kebab', 'kebob'], 'empire state building': ['empire state'],
                              'mount kilimanjaro': ['kilimanjaro'], 'sashimi': ['sashimi'], 'sushi': ['sushi'],
                              'tacos': ['tacos', 'taco'], 'sydney opera house': ['sydney opera house'],
                              'rolls-royce': ['rolls-royce', 'rolls royce', 'rr'], 'jasmine': ['jasmine'],
                              'microsoft': ['microsoft'], 'empanadas': ['empanadas', 'empanada'],
                              'shaolin temple': ['shaolin temple', 'shaolin monastery'],
                              'godzilla': ['godzilla', 'king monsters'],
                              'pad thai': ['pad thai', 'phat thai', 'phad thai'],
                              'london eye': ['millennium wheel', 'london eye'],
                              'sensoji temple': ['sensoji temple', 'sensō-ji'],
                              'edinburgh castle': ['edinburgh'],
                              'summer palace': ['summer', 'kunming lake'], 'sony': ['sony'],
                              'porsche': ['porsche'], 'pinocchio': ['pinocchio'], 'gandalf': ['gandalf'],
                              'palace of westminster': ['palace westminster', 'houses parliament'],
                              'big ben': ['big ben'], 'angkor wat': ['angkor wat'],
                              'general electric': ['general electric', 'ge'],
                              'hot pot': ['hot pot', 'hotpot', 'soup-food', 'steamboat'],
                              'santa claus': ['santa claus', 'father christmas'], 'laksa': ['laksa'],
                              'burger king': ['king'], 'colosseum': ['colosseum', 'colosseo'],
                              'shanghai tower': ['shanghai'], 'bose': ['bose'],
                              'homer simpson': ['homer simpson', 'homer jay simpson'],
                              'taipei 101': ['taipei 101', 'taipei world financial center'], 'kfc': ['kfc'],
                              'razer': ['razer'], 'satay': ['satay', 'sate'],
                              'mcdonald': ['mcdonald', "mcdonald's", 'mcd', 'mcdo'],
                              'hoover dam': ['hoover dam', 'boulder dam'], 'mickey mouse': ['mickey mouse'],
                              'sydney harbour bridge': ['sydney harbour', 'coathanger'],
                              'leaning tower of pisa': ['leaning pisa', 'torre pendente di pisa'],
                              'cartier': ['cartier'], 'smoked salmon': ['smoked salmon']}
    num_pred = 0

    # 为knowcap生成结果
    print("Test num: "+str(len(knowcap_data)))
    with torch.no_grad():
        for i, item in tqdm(enumerate(knowcap_data)):
            image_path = os.path.join(knowcap_img_dir, item["image"])
            img = Image.open(image_path)
            patch_img = patch_resize_transform(img).unsqueeze(0).to(device)
            all_tokens = model.generate_caption_batchbs(patch_img)
            if config.model == 'OFA':
                gen = all_tokens[0].unsqueeze(0)
                caption = model.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
            elif config.model == 'BLIP':
                caption = model.tokenizer.decode(all_tokens[0], skip_special_tokens=True)
                caption = caption[len(model.prompt):]
            elif config.model == 'GIT':
                gen = all_tokens[0].unsqueeze(0)
                caption = model.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
            gen_pycoco[i] = [{'image_id': item["image"], 'id': i, 'caption': caption}]
            # 是否预测出关键词
            category = item["image"].split('/')[0]
            keywords = categories_to_keywords[category]
            keywords_recog = []
            for keyword in keywords:
                keywords_recog += keyword.split(' ')
            keywords_recog = set(keywords_recog)
            for keyword in keywords_recog:
                if keyword in caption:
                    num_pred += 1
                    break
    json.dump(gen_pycoco, open(gen_pycoco_path, 'w'), ensure_ascii=False)

    if True:  # 一些官方的ckpts会出现多余字符需要后处理
        pycoco = json.load(open(gen_pycoco_path, 'r'))
        for k, v in pycoco.items():
            caption_origin = v[0]["caption"]
            caption_new = caption_origin.replace(')', '').replace('\\', '').replace('}', '').replace(']', '').strip()
            v[0]["caption"] = caption_new
        json.dump(pycoco, open(gen_pycoco_path, 'w'))

    # 计算标准指标
    gen_pycoco = json.load(open(gen_pycoco_path, 'r'))
    ref_pycoco = json.load(open(ref_pycoco_path, 'r'))
    ref_pycoco = {int(k): v for k, v in ref_pycoco.items()}  # json读取时key类型为str，在计算SPICE时会出现问题
    gen_pycoco = {int(k): v for k, v in gen_pycoco.items()}
    cocoEval = COCOEvalCap('diy', 'diy')
    pycoco_results = cocoEval.evaluate_diy(ref_pycoco, gen_pycoco)
    pycoco_results_return = {}
    for k, v in pycoco_results.items():
        pycoco_results_return[k+'_knowcap'] = v
    print(pycoco_results)

    # 计算概念覆盖度
    recog_acc = num_pred / len(knowcap_data)
    print(recog_acc)
    return pycoco_results_return, {"recog acc": recog_acc}


if __name__ == '__main__':
    # model
    model = construct_model(config).to(device)
    if config.id != 'test':
        log_path = config.log_dir.format(config.id)
        trained_model_path = log_path + '/model/model_' + str(config.step) + '.pt'
        model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    cal_knowcap(model, config.step, mode='test', unseen=False)
    cal_knowcap(model, config.step, mode='test', unseen=True)

