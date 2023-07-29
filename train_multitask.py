# training with K-Replay
import yaml
import json
import torch
import random
import numpy as np

import time
import torch.nn as nn

from config import config
from data_load import data_load_rwc

from utils.import_models import construct_model
from utils.loss import Cross_Entropy, Loss_SCST_OFA, Sent_Level_Concept_Coverage, Loss_Params_Regular, Loss_KD
from utils.log import Log_Writer, train_print
from utils.eval import generate_captions, eval_pycoco
from utils.optimizer_tools import adjust_weight, adjust_lr, cal_fisher_coco, cal_fisher_downtask_mask, adjust_mask, model_grad_mask, RecAdam, cal_fisher_downtask, ratio_dataset
from utils.vocab import Vocabulary
from test_knowcap import cal_knowcap
from models.OFA.ofa import OFA
from models.BLIP.blip import blip_decoder
from models.GIT.git import GIT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机种子
seed = config.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# log
writer = Log_Writer(config)
global_step = 0
loss_avg = 0
loss_ce_avg = 0
loss_rwc_avg = 0
mt_weight = config.multitask_weight
kd_weight = config.knowdistill_weight

train_mix = config.train_mix    # 用于训练的数据集，既可以是混合的，也可以是coco单独的
if config.data_ratio != 1.0:    # 可调整coco和其他的比例
    train_mix_data_new = ratio_dataset(train_mix, config.data_ratio)
    train_mix = './data/train_mix_cc12m_keyword_'+str(config.data_ratio)+'.json'
    json.dump(train_mix_data_new, open(train_mix, 'w'))

data_mode = config.data_mode    # 和train_mix配合使用，决定训练数据和模式，mix|single
method = config.method  # 比较的各种方法
model_type = config.model
# data_loader
train_loader = data_load_rwc(config, train_mix, 'train')

# model
model = construct_model(config).to(device)
if method == 'XEdistill':
    if model_type == 'OFA':
        model_t = OFA(config, distill_model=True)
    elif model_type == 'BLIP':
        argst = yaml.load(open(config.config_blip_t, 'r'), Loader=yaml.Loader)
        model_t = blip_decoder(pretrained=argst['pretrained'], config=config, image_size=argst['image_size'],
                               vit=argst['vit'],
                               vit_grad_ckpt=argst['vit_grad_ckpt'], vit_ckpt_layer=argst['vit_ckpt_layer'],
                               prompt=argst['prompt'])
    elif model_type == 'GIT':
        model_t = GIT(config, distill_model=True)
    model_t = model_t.to(device)
    loss_distill = Loss_KD(config.KD_temperature)
if method == 'Adapter':     # Adapter使得只有小部分模型参数参与训练
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            if 'adapter_ln1' in name or 'adapter_ln2' in name:
                p.requires_grad = True
                print(name)
            else:
                p.requires_grad = False

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
if method == 'RecAdam':    # Recall and Learn利用优化器做正则化
    pretrain_params = []
    for name, p in model.named_parameters():
        pretrain_params.append(p)
    optimizer = RecAdam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9, anneal_k=1.0, anneal_t0=100, pretrain_params=pretrain_params)

# loss
loss_cov = Sent_Level_Concept_Coverage()
loss_fn = Cross_Entropy(label_smoothing=config.label_smoothing)

if method == 'child-tuning':    # Child-tuning和EWC与参数在对应任务梯度有关，因此先计算相关梯度
    grads_mask_coco = cal_fisher_downtask_mask(config, model)
    # grads_mask_coco = adjust_mask(grads_mask_coco)
elif method == 'k-tuning':
    grads_mask_knowledge = cal_fisher_downtask_mask(config, model)  # 找到和知识相关的参数
elif method == 'EWC':
    params_fisher = cal_fisher_downtask(config, model)
    params_init = dict()
    for name, params in model.named_parameters():
        if params.requires_grad == True:
            params_init[name] = params
    loss_params_regular = Loss_Params_Regular(params_init, params_fisher)

if config.step != 0:
    log_path = config.log_dir.format(config.ckpts_id)
    trained_model_path = log_path + '/model/model_' + str(config.step) + '.pt'
    model.load_state_dict(torch.load(trained_model_path))
    global_step = config.step

for epoch in range(config.epochs):
    if global_step == 800:
        break
    model.train()
    totel_step = len(train_loader)
    epoch_time = time.time()
    step_time = time.time()

    optimizer = adjust_lr(optimizer, epoch)
    for step, (image_feature, cap, att_mask, cap_len, labels, data_item) in enumerate(train_loader):

        data_mode = config.data_mode
        global_step += 1
        optimizer.zero_grad()

        patch_image = image_feature['patch_image']
        patch_image = patch_image.to(device)
        cap = cap.to(device)
        cap_len = cap_len.to(device)
        labels = labels.to(device)
        att_mask = att_mask.to(device)

        if labels.sum().item() == 0:
            data_mode = 'single'

        if data_mode == 'mix':  # 找到其中rwconcept的样本，构建伪<image, caption> pair进行训练
            index_rwc = torch.nonzero(labels==1).squeeze().long()
            if index_rwc.shape == torch.Size([]):
                index_rwc = index_rwc.unsqueeze(0)
            index_coco = torch.nonzero(labels==0).squeeze(dim=1).long()
            # 保存原caption以作为label
            cap_rwc_label = cap[index_rwc]
            # 为这些样本用当前模型生成伪caption
            if index_rwc.shape != torch.Size([0]):
                with torch.no_grad():
                    patch_image_rwc = patch_image[index_rwc]
                    all_tokens, all_logprob = model.greedy_search(patch_image_rwc, 'max')
                    cap_new = []
                    att_mask_new = []
                    cap_len_new = []
                    for cap_id in all_tokens:
                        cap_len_g = cap_id.shape[0]
                        if cap_len_g < config.fixed_len:
                            if model_type == 'OFA':
                                cap_id = torch.cat([cap_id, torch.ones([config.fixed_len - cap_len_g]).to(device)], dim=0)
                            elif model_type == 'BLIP':
                                cap_id = torch.cat([cap_id, torch.zeros([config.fixed_len - cap_len_g]).to(device)], dim=0)
                            elif model_type == 'GIT':
                                cap_id = torch.cat([cap_id, torch.zeros([config.fixed_len - cap_len_g]).to(device)], dim=0)
                            att_mask_g = torch.cat([torch.ones([cap_len_g]).to(device), torch.zeros([config.fixed_len - cap_len_g]).to(device)], dim=0)
                        else:
                            cap_id = cap_id[:config.fixed_len]
                            cap_len_g = config.fixed_len
                            att_mask_g = torch.ones(cap_id.shape).to(device)
                        cap_new.append(cap_id)
                        att_mask_new.append(att_mask_g)
                        cap_len_new.append(cap_len_g)
                    cap_new = torch.stack(cap_new, dim=0).long()
                    att_mask_new = torch.stack(att_mask_new, dim=0).long()
                    cap_len_new = torch.Tensor(cap_len_new).int()
                # 将伪caption放回原数据中一起进行forward
                cap[index_rwc] = cap_new.to(device)
                att_mask[index_rwc] = att_mask_new.to(device)
                cap_len[index_rwc] = cap_len_new.to(device)
                # 知识蒸馏，用teacher进行一次前向传播获得logit
                if method == 'XEdistill':
                    with torch.no_grad():
                        logit_t = model_t(patch_image[index_rwc], cap[index_rwc], att_mask[index_rwc], cap_len[index_rwc])

        logit = model(patch_image, cap, att_mask, cap_len)
        if data_mode == 'single':
            loss = loss_fn(logit, cap, cap_len)
            loss_avg += loss.item()
        elif data_mode == 'mix':
            loss_ce = loss_fn(logit[index_coco], cap[index_coco], cap_len[index_coco])
            loss_rwc = loss_cov(logit[index_rwc], cap_rwc_label, cap_len[index_rwc], model_type)
            loss = loss_ce + mt_weight * loss_rwc
            if method == 'XEdistill':
                loss_kd = loss_distill(logit[index_rwc], logit_t, cap_len[index_rwc])
                loss += kd_weight*loss_kd
            loss_ce_avg += loss_ce.item()
            loss_rwc_avg += loss_rwc.item()
            loss_avg += loss.item()

        if method == 'EWC':
            loss = loss + loss_params_regular(model)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if method == 'child-tuning':
            model_grad_mask(model, grads_mask_coco)
        optimizer.step()

        if global_step % config.save_loss_freq == 0:
            writer.write_tensorboard('loss', loss_avg/config.save_loss_freq, global_step)
            loss_avg = 0
            if data_mode == 'mix':
                writer.write_tensorboard('loss_ce', loss_ce_avg/config.save_loss_freq, global_step)
                writer.write_tensorboard('loss_rwc', loss_rwc_avg/config.save_loss_freq, global_step)
                loss_ce_avg = 0
                loss_rwc_avg = 0

        train_print(loss.item(), step, totel_step, epoch, time.time() - step_time, time.time() - epoch_time)
        step_time = time.time()

        if global_step % config.save_model_freq == 0:
            print("Evaluating...")

            # 保存模型
            if global_step % 100 == 0:
                writer.save_model(model, global_step)

            # validation
            model.eval()
            with torch.no_grad():
                gen_pycoco_path = generate_captions(config, model, global_step, 'val')
            pycoco_results = eval_pycoco(config, gen_pycoco_path, 'val')
            pycoco_results_knowcap, acc = cal_knowcap(model, global_step)
            writer.write_metrics(pycoco_results, global_step)
            writer.write_metrics(pycoco_results_knowcap, global_step)
            writer.write_metrics(acc, global_step)

            model.train()

        if global_step == 800:
            break