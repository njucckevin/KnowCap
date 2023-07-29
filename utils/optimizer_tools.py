import random

import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_load import data_load_rwc, data_load, data_load_rwc_EWC
from utils.loss import Cross_Entropy_Keyword, Cross_Entropy
from torch.optim import Optimizer
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ratio_dataset(train_mix, data_ratio):
    train_mix_data = json.load(open(train_mix, 'r'))
    data_coco = []
    data_else = []
    for item in train_mix_data:
        if item['data'] == 'coco':
            data_coco.append(item)
        elif item['data'] == 'cc12m':
            data_else.append(item)

    random.shuffle(data_else)
    data_else = data_else[:int(len(data_else)*data_ratio)]
    print("New else num: "+str(len(data_else)))
    return data_coco+data_else


def adjust_lr(optimizer, epoch):
    if epoch >= 9 and epoch % 3 == 0:
        print('*** adjust learning rate ***')
        for p in optimizer.param_groups:
            p['lr'] = p['lr'] * 0.8
    return optimizer


def adjust_weight(weight_init, global_step, method='linear'):
    if method == 'linear':
        t_max = 500
        if global_step < 500:
            return weight_init*(global_step/500)
        else:
            return weight_init
    elif method == 'sigmoid_annealing':
        t0 = 250
        k = 0.01
        lambda_t = 1 / (1+math.exp(-k*(global_step-t0)))
        return lambda_t*weight_init


def cal_grads_mask_overlap(grads_mask_coco, grads_mask_rwc):
    a_true = 0
    b_true = 0
    ab_true = 0
    for k in grads_mask_coco:
        a_mask = grads_mask_coco[k]
        b_mask = grads_mask_rwc[k]
        a_true += a_mask.sum().item()
        b_true += b_mask.sum().item()
        ab_mask = a_mask*b_mask
        ab_true += ab_mask.sum().item()


def layers_mask(model, grads_mask):
    for name, params in model.named_parameters():
        if params.requires_grad == True:
            if grads_mask[name].item() == False:
                params.requires_grad = False
            else:
                print(name)


def count_layers_grad(model):
    num = 0
    for name, params in model.named_parameters():
        if params.requires_grad == True:
            num += 1
    print("num of layer for optim: "+str(num))


def adjust_mask(grads_mask_rwc):

    # 翻转mask
    for k in grads_mask_rwc:
        grads_mask_rwc[k] = ~(grads_mask_rwc[k].type(torch.bool))
    return grads_mask_rwc

    """
    # 在rwcmask中但不在cocomask中
    for k in grads_mask_rwc:
        grads_mask_rwc[k] = grads_mask_rwc[k]^grads_mask_rwc[k]*grads_mask_coco[k]
    return grads_mask_rwc
    """
    """
    # 随机mask
    grads_mask = dict()
    for k in grads_mask_rwc:
        grads_mask[k] = (torch.rand(grads_mask_rwc[k].shape).to(device) > 0.95)
    return grads_mask
    """


def cal_fisher_coco(config, model, loss_fn):
    params_grad = dict()
    model.train()
    for name, params in model.named_parameters():
        if params.requires_grad == True:
            params_grad[name] = params.new_zeros(params.size())
    
    loss_avg = 0
    N = 100
    data_loader = data_load(config, config.train, 'train')
    # N = len(data_loader)
    for step, (image_feature, cap, att_mask, cap_len, refs) in tqdm(enumerate(data_loader)):
        if step == N:
            break
        patch_image = image_feature['patch_image']
        patch_image = patch_image.to(device)
        cap = cap.to(device)
        cap_len = cap_len.to(device)
        att_mask = att_mask.to(device)

        logit = model(patch_image, cap, att_mask, cap_len)
        loss = loss_fn(logit, cap, cap_len)

        loss.backward()
        loss_avg += loss.item()

        for name, params in model.named_parameters():
            if params.requires_grad == True:
                if params.grad != None:
                    torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                    params_grad[name] += (params.grad ** 2) / N

        model.zero_grad()

    print("loss avg: "+str(loss_avg/N))
    """
    for k in params_grad:
        params_grad[k] = params_grad[k].mean()
    layers_list = []
    for k, v in params_grad.items():
        layers_list.append(v.unsqueeze(0))
    r = torch.cat(layers_list, dim=0)
    r = r.cpu().numpy()
    print("num of layers: "+str(r.size))
    plt.plot(r)
    plt.savefig('./grad_layers_{}'.format(g_step), dpi=1000)
    print("params_grad save")
    """
    params_list = []
    for k, v in tqdm(params_grad.items()):
        v = v.view(-1)
        params_list.append(v)
    r = torch.cat(params_list, dim=0)
    r = r.cpu().numpy()

    print("num of parameters: "+str(r.size))
    #plt.plot(r)
    #plt.savefig('./grad_20_{}'.format(g_step), dpi=1000)
    #print("params_grad save")
    
    reserve_p = 0.05
    polar = np.percentile(r, (1-reserve_p)*100)
    print("polar: "+str(polar))
    
    grads_mask = dict()
    for k in params_grad:
        grads_mask[k] = params_grad[k] >= polar
        # grads_mask[k] = (torch.rand(params_grad[k].shape).to(device) > 0.95)
    
    return grads_mask


def cal_fisher_downtask_mask(config, model):
    loss_fn = Cross_Entropy()
    rwconcept_data_path = '/home/data_ti4_c/chengkz/ofa/data/train_all.json'
    data_loader = data_load_rwc(config, rwconcept_data_path, 'train')
    params_grad = dict()
    model.train()

    for name, params in model.named_parameters():
        if params.requires_grad == True:
            params_grad[name] = params.new_zeros(params.size())

    N = 100
    # N = len(data_loader)
    loss_avg = 0
    for step, (image_feature, cap, att_mask, cap_len, labels) in tqdm(enumerate(data_loader)):
        if step == N:
            break
        patch_image = image_feature['patch_image']
        patch_image = patch_image.to(device)
        cap = cap.to(device)
        cap_len = cap_len.to(device)
        att_mask = att_mask.to(device)

        logit = model(patch_image, cap, att_mask, cap_len)
        loss = loss_fn(logit, cap, cap_len)

        loss.backward()
        loss_avg += loss.item()

        for name, params in model.named_parameters():
            if params.requires_grad == True:
                if params.grad != None:
                    torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                    params_grad[name] += (params.grad ** 2) / N

        model.zero_grad()

    params_list = []
    for k, v in tqdm(params_grad.items()):
        v = v.view(-1)
        params_list.append(v)
    r = torch.cat(params_list, dim=0)
    r = r.cpu().numpy()
    print("num of parameters: " + str(r.size))
    reserve_p = 0.05
    polar = np.percentile(r, (1 - reserve_p) * 100)
    print("polar: " + str(polar))

    grads_mask = dict()
    for k in params_grad:
        grads_mask[k] = params_grad[k] >= polar

    return grads_mask


def cal_fisher_downtask(config, model):
    loss_fn = Cross_Entropy()
    rwconcept_data_path = '/home/data_ti4_c/chengkz/ofa/data/data_cc12m_SelectForReplay.json'
    data_loader = data_load_rwc_EWC(config, rwconcept_data_path, 'train')
    params_grad = dict()
    model.train()

    for name, params in model.named_parameters():
        if params.requires_grad == True:
            params_grad[name] = params.new_zeros(params.size())

    N = 100
    # N = len(data_loader)
    loss_avg = 0
    for step, (image_feature, cap, att_mask, cap_len, labels, if_keyword) in tqdm(enumerate(data_loader)):
        if step == N:
            break
        patch_image = image_feature['patch_image']
        patch_image = patch_image.to(device)
        cap = cap.to(device)
        cap_len = cap_len.to(device)
        att_mask = att_mask.to(device)
        if_keyword = if_keyword.to(device)

        logit = model(patch_image, cap, att_mask, cap_len)
        loss = loss_fn(logit, cap, cap_len)

        loss.backward()
        loss_avg += loss.item()

        for name, params in model.named_parameters():
            if params.requires_grad == True:
                if params.grad != None:
                    torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                    params_grad[name] += (params.grad ** 2) / N

        model.zero_grad()

    return params_grad


def model_grad_mask(model, grads_mask):
    for name, params in model.named_parameters():
        if params.requires_grad == True:
            if params.grad != None:
                params.grad = params.grad*grads_mask[name]


def anneal_function(function, step, k, t0, weight):
    if function == 'sigmoid':
        return float(1 / (1 + np.exp(-k * (step - t0)))) * weight
    elif function == 'linear':
        return min(1, step / t0) * weight
    elif function == 'constant':
        return weight
    else:
        ValueError


class RecAdam(Optimizer):
    """ Implementation of RecAdam optimizer, a variant of Adam optimizer.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        anneal_fun (str): a hyperparam for the anneal function, decide the function of the curve. Default 'sigmoid'.
        anneal_k (float): a hyperparam for the anneal function, decide the slop of the curve. Choice: [0.05, 0.1, 0.2, 0.5, 1]
        anneal_t0 (float): a hyperparam for the anneal function, decide the middle point of the curve. Choice: [100, 250, 500, 1000]
        anneal_w (float): a hyperparam for the anneal function, decide the scale of the curve. Default 1.0.
        pretrain_cof (float): the coefficient of the quadratic penalty. Default 5000.0.
        pretrain_params (list of tensors): the corresponding group of params in the pretrained model.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True,
                 anneal_fun='sigmoid', anneal_k=0, anneal_t0=0, anneal_w=1.0, pretrain_cof=5000.0, pretrain_params=None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias,
                        anneal_fun=anneal_fun, anneal_k=anneal_k, anneal_t0=anneal_t0, anneal_w=anneal_w,
                        pretrain_cof=pretrain_cof, pretrain_params=pretrain_params)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p, pp in zip(group["params"], group["pretrain_params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # With RecAdam method, the optimization objective is
                # Loss = lambda(t)*Loss_T + (1-lambda(t))*Loss_S
                # Loss = lambda(t)*Loss_T + (1-lambda(t))*\gamma/2*\sum((\theta_i-\theta_i^*)^2)
                if group['anneal_w'] > 0.0:
                    # We calculate the lambda as the annealing function
                    anneal_lambda = anneal_function(group['anneal_fun'], state["step"], group['anneal_k'],
                                                    group['anneal_t0'], group['anneal_w'])
                    # anneal_lambda = 0.9
                    assert anneal_lambda <= group['anneal_w']
                    # The loss of the target task is multiplied by lambda(t)
                    p.data.addcdiv_(-step_size * anneal_lambda, exp_avg, denom)
                    # Add the quadratic penalty to simulate the pretraining tasks
                    p.data.add_(-group["lr"] * (group['anneal_w'] - anneal_lambda) * group["pretrain_cof"], p.data - pp.data)
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss
