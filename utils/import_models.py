# 根据命令行构建模型

import os
import yaml
from pathlib import Path
from models.Transformer.transformer import Transformer_Cap
from models.OFA.ofa import OFA
from models.BLIP.blip import blip_decoder
from models.GIT.git import GIT


def construct_model(config):
    if config.model == 'Transformer':
        model = Transformer_Cap(config)
    elif config.model == 'OFA':
        model = OFA(config)
    elif config.model == 'BLIP':
        args = yaml.load(open(config.config_blip, 'r'), Loader=yaml.Loader)
        model = blip_decoder(pretrained='/home/chengkz/.cache/torch/hub/checkpoints/model_large_trainedenc.pth', config=config, image_size=args['image_size'],
                             vit=args['vit'],
                             vit_grad_ckpt=args['vit_grad_ckpt'], vit_ckpt_layer=args['vit_ckpt_layer'],
                             prompt=args['prompt'])
    elif config.model == 'GIT':
        model = GIT(config)
    else:
        print("model "+str(config.model)+" not found")
        return None
    return model