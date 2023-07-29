import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from evaluation import Cider
import numpy as np
import pickle
import json
from transformers.models.ofa.tokenization_ofa import OFATokenizer
import torch.nn.functional as F

from config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 用于XEdistill
class Loss_KD(nn.Module):

    def __init__(self, KD_T=8):
        super(Loss_KD, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = KD_T

    def forward(self, logit, logit_teacher, cap_len):
        prob = self.softmax(logit / self.temperature)
        prob_teacher = self.softmax(logit_teacher / self.temperature)

        pred = pack_padded_sequence(prob, cap_len.cpu(), batch_first=True, enforce_sorted=False)[0]
        target = pack_padded_sequence(prob_teacher, cap_len.cpu(), batch_first=True, enforce_sorted=False)[0]

        loss_kl = F.kl_div(pred.log(), target, reduction='sum') / logit.shape[0]
        return loss_kl


# Label Smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.linear_combination(loss / n, nll, self.epsilon)


class Cross_Entropy(nn.Module):
    # 序列形式的交叉熵
    def __init__(self, label_smoothing=0.0):
        super(Cross_Entropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss().to(device)
        self.ce_ls = LabelSmoothingCrossEntropy(epsilon=label_smoothing).to(device)

    def forward(self, logit, cap, cap_len):
        target = cap[:, 1:]
        cap_len = cap_len - 1

        target = pack_padded_sequence(target, cap_len.cpu(), batch_first=True, enforce_sorted=False)[0]
        logit = pack_padded_sequence(logit, cap_len.cpu(), batch_first=True, enforce_sorted=False)[0]

        # cross_entropy
        if self.label_smoothing > 0:
            loss_ce = self.ce_ls(logit, target)
        else:
            loss_ce = self.ce(logit, target)

        return loss_ce


# 只计算知识关键词的交叉熵，用于寻找和知识相关的参数
class Cross_Entropy_Keyword(nn.Module):
    # 序列形式的交叉熵
    def __init__(self):
        super(Cross_Entropy_Keyword, self).__init__()
        self.ce = nn.CrossEntropyLoss().to(device)

    def forward(self, logit, cap, cap_len, if_keyword):
        target = cap[:, 1:]
        if_keyword = if_keyword[:, 1:] > 0
        logit = logit[:, :-1]
        cap_len = cap_len - 1

        target = target[if_keyword]
        logit = logit[if_keyword]

        # cross_entropy
        loss_ce = self.ce(logit, target)

        return loss_ce


# K-Replay的核心损失函数，预测知识关键词
class Sent_Level_Concept_Coverage(nn.Module):
    def __init__(self):
        super(Sent_Level_Concept_Coverage, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, logit_rwc, cap_rwc_label, cap_len_rwc, model_type):
        softmax_rwc = self.softmax(logit_rwc)
        loss_cov = torch.zeros(cap_len_rwc.shape[0]).to(device)
        loss_rep = torch.zeros(cap_len_rwc.shape[0]).to(device)
        for i in range(cap_len_rwc.shape[0]):
            softmax_sen = softmax_rwc[i][:cap_len_rwc[i].item()]
            softmax_agg = softmax_sen.sum(dim=0)
            sigmoid_agg = self.sigmoid(softmax_agg)
            if model_type == 'OFA':
                label = cap_rwc_label[i][cap_rwc_label[i]>2]
            elif model_type == 'BLIP':
                label = cap_rwc_label[i][(cap_rwc_label[i]!=0) & (cap_rwc_label[i]!=102) & (cap_rwc_label[i]!=30522)
                                         & (cap_rwc_label[i]!=1037) & (cap_rwc_label[i]!=3861) & (cap_rwc_label[i]!=1997)]
            elif model_type == 'GIT':
                label = cap_rwc_label[i][(cap_rwc_label[i]!=0) & (cap_rwc_label[i]!=101) & (cap_rwc_label[i]!=102)]
            prob = sigmoid_agg[label]
            log_prob = -torch.log(prob).mean()
            loss_cov[i] = log_prob
            prob_softmax = softmax_agg[label]
            prob_pow = torch.pow(1-prob_softmax, 2).mean()
            loss_rep[i] = prob_pow
        loss_cov = loss_cov.mean()
        loss_rep = loss_rep.mean()
        loss_rwc = loss_cov+loss_rep
        return loss_rwc


class Loss_Params_Regular(nn.Module):
    def __init__(self, params_init, params_fisher):
        super(Loss_Params_Regular, self).__init__()
        self.params_init = params_init
        self.params_fisher = params_fisher
        self.gamma = 50000

    def forward(self, model):
        loss = 0
        for name, params in model.named_parameters():
            if params.requires_grad == True:
                loss_p = 0.5 * self.gamma * self.params_fisher[name] * torch.pow(params-self.params_init[name], 2)
                loss += loss_p.sum()
        return loss


class Loss_SCST(nn.Module):

    def __init__(self, config):
        super(Loss_SCST, self).__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.beam_num = config.beam_num
        self.vocab = pickle.load(open(config.vocab, 'rb'))
        self.train = json.load(open(config.train, 'r'))
        self.cider_texts = {i: [' '.join(item['caption'])] for i, item in enumerate(self.train)}
        self.cider_train = Cider(self.cider_texts)

    def vanilla_scst(self, all_tokens, all_tokens_greedy, all_logprob, refs):
        # vanilla scst: 多项式采样beam_num个，greedy作为baseline
        # 首先将greedy和ref复制beam_num倍
        gen_num = len(all_tokens)
        all_tokens_greedy_beam = []
        for item in all_tokens_greedy:
            all_tokens_greedy_beam.extend([item for i in range(self.beam_num)])
        refs_beam = []
        for item in refs:
            refs_beam.extend([item for i in range(self.beam_num)])

        # 整理采样、greedy和ref计算指标
        caps_gen = {i: [self.vocab.idList_to_sent(item)] for i, item in enumerate(all_tokens)}
        caps_gen_greedy = {i: [self.vocab.idList_to_sent(item)] for i, item in enumerate(all_tokens_greedy_beam)}
        caps_gt = {i: item for i, item in enumerate(refs_beam)}
        reward = self.cider_train.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
        reward = torch.from_numpy(reward).to(device).view(gen_num)
        reward_baseline = self.cider_train.compute_score(caps_gt, caps_gen_greedy)[1].astype(np.float32)
        reward_baseline = torch.from_numpy(reward_baseline).to(device).view(gen_num)

        # 对采样结果的log_prob补齐
        all_logprob_pad = []
        for logprob in all_logprob:
            logprob = torch.cat([logprob, logprob.new([0 for i in range(self.config.fixed_len - logprob.shape[0])])], dim=0)
            all_logprob_pad.append(logprob.unsqueeze(0))
        all_logprob_pad = torch.cat(all_logprob_pad, dim=0)

        # 计算损失
        loss = -torch.mean(all_logprob_pad, -1) * (reward - reward_baseline)
        loss = loss.mean()

        # 计算训练reward
        reward_train = reward.mean()

        return loss, reward_train


class Loss_SCST_OFA(nn.Module):

    def __init__(self, config):
        super(Loss_SCST_OFA, self).__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.beam_num = config.beam_num
        self.tokenizer = OFATokenizer.from_pretrained(self.config.ofa_ckpts)
        self.train = json.load(open(config.train, 'r'))
        self.cider_texts = {i: [' '.join(item['caption'])] for i, item in enumerate(self.train)}
        self.cider_train = Cider(self.cider_texts)

    def vanilla_scst(self, all_tokens, all_tokens_greedy, all_logprob, refs):
        # vanilla scst: 多项式采样beam_num个，greedy作为baseline
        # 首先将greedy和ref复制beam_num倍
        gen_num = len(all_tokens)
        all_tokens_greedy_beam = []
        for item in all_tokens_greedy:
            all_tokens_greedy_beam.extend([item for i in range(self.beam_num)])
        refs_beam = []
        for item in refs:
            refs_beam.extend([item for i in range(self.beam_num)])

        # 整理采样、greedy和ref计算指标
        caps_gen = {i: [self.tokenizer.batch_decode(item.unsqueeze(0), skip_special_tokens=True)[0].strip()] for i, item in enumerate(all_tokens)}
        caps_gen_greedy = {i: [self.tokenizer.batch_decode(item.unsqueeze(0), skip_special_tokens=True)[0].strip()] for i, item in enumerate(all_tokens_greedy_beam)}
        caps_gt = {i: item for i, item in enumerate(refs_beam)}
        reward = self.cider_train.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
        reward = torch.from_numpy(reward).to(device).view(gen_num)
        reward_baseline = self.cider_train.compute_score(caps_gt, caps_gen_greedy)[1].astype(np.float32)
        reward_baseline = torch.from_numpy(reward_baseline).to(device).view(gen_num)

        # 对采样结果的log_prob补齐
        all_logprob_pad = []
        for logprob in all_logprob:
            logprob = torch.cat([logprob, logprob.new([0 for i in range(self.config.fixed_len - logprob.shape[0])])], dim=0)
            all_logprob_pad.append(logprob.unsqueeze(0))
        all_logprob_pad = torch.cat(all_logprob_pad, dim=0)

        # 计算损失
        loss = -torch.mean(all_logprob_pad, -1) * (reward - reward_baseline)
        loss = loss.mean()

        # 计算训练reward
        reward_train = reward.mean()

        return loss, reward_train


