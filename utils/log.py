# 训练日志
# 写tensorboard
# 保存模型

import time
import os
import json
import sys
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter


def train_print(loss, step, total_step, epoch, step_time, epoch_time):
    epoch_time = time.localtime(epoch_time)
    min = epoch_time.tm_min
    sec = epoch_time.tm_sec
    print(f"\rloss:{format(loss, '.8f')} |"
          f"step: {step}/{total_step} |"
          f"epoch: {epoch} |"
          f"step time:{format(step_time, '.2f')}secs |",
          f"epoch time: {min}min {sec}sec", end='')


class Log_Writer():

    def __init__(self, config):
        super(Log_Writer, self).__init__()

        print("Creating Log dir...")
        self.log_path = config.log_dir.format(config.id)
        if not os.path.exists(self.log_path):    # 创建log路径
            os.makedirs(self.log_path)

        para_path = os.path.join(self.log_path, 'para.json')     # 保存命令行参数
        with open(para_path, 'w') as f:
            json.dump(sys.argv, f)
        shutil.copy('./config.py', self.log_path)    # 保存config参数

        self.writer = SummaryWriter(self.log_path)   # tensorboard writer

    def write_tensorboard(self, scalar_name, scalar, step):
        self.writer.add_scalar(scalar_name, scalar, step)

    def write_metrics(self, pycoco_results, step):
        # metrics_list = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
        for metric in pycoco_results:
            self.write_tensorboard(metric, pycoco_results[metric], step)

    def save_model(self, model, global_step):
        model_path = os.path.join(self.log_path, 'model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_path = os.path.join(model_path, f'model_{global_step}.pt')
        torch.save(model.state_dict(), save_path)

