## Beyond Generic: Enhancing Image Captioning with Real-World Knowledge using Vision-Language Pre-Training Model
This repo provides the source code & data of our paper: [Beyond Generic: Enhancing Image Captioning with Real-World Knowledge using Vision-Language Pre-Training Model. (ACMMM 23)](https://arxiv.org/abs/2308.01126)
```
@misc{cheng2023generic,
      title={Beyond Generic: Enhancing Image Captioning with Real-World Knowledge using Vision-Language Pre-Training Model}, 
      author={Kanzhi Cheng and Wenpo Song and Zheng Ma and Wenhao Zhu and Zixuan Zhu and Jianbing Zhang},
      year={2023},
      eprint={2308.01126},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
### Code Structure
***
````
├── config.py	# config
├── data	# coco data & knowcap data
│   ├── data_cc12m_SelectForRreplay.json
│   ├── dataset_coco.json
│   ├── test.json
│   ├── train.json
│   ├── val.json
│   ├── knowcap_240.json
│   ├── knowcap_240_test.json
│   ├── knowcap_240_test_unseen.json
│   ├── knowcap_240_val.json
│   ├── train_mix_32000.json
│   └── ...
├── data_load.py	# dataloader
├── test.py	    # evaluation on coco
├── test_knowcap.py	    # evaluation on knowcap
├── models	# models (OFA,BLIP,GIT)
│   ├── OFA
│   ├── BLIP
│   └── GIT
├── train_multitask.py      # K-Replay training
└── utils	# support codes & tools
    ├── beamsearch.py	# beamsearch
    ├── cc12m.py	# filter relay data from cc12m
    ├── convert_ofa.py	# ckpts convert
    ├── eval.py		# generate captions & calculate metrics
    ├── import_models.py
    ├── log.py
    ├── loss.py		# loss function of K-Replay
    ├── optimizer_tools.py
    └── prepro_data.py   # construct the data in ./data
````
### KnowCap Dataset
***
KnowCap is a new dataset for the evaluation of knowledge-enhanced image captioning, containing 1424 images and 4156 reference descriptions
carefully written by human annotators.

![](https://beyondgeneric.s3.amazonaws.com/knowcap.png)

Download the images and annotations of [KnowCap](https://beyondgeneric.s3.amazonaws.com/KnowCap.zip).
### Preparing Data&Model
***
#### Step1：
Download the images of:
 * [COCO2014](https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/data/README.md)
 * [KnowCap](https://beyondgeneric.s3.amazonaws.com/KnowCap.zip)
 * [Replay images selected from cc12m](https://beyondgeneric.s3.amazonaws.com/cc12m_replay.zip)
#### Step2：
`prepro_data.py`, Collate and split coco and knowcap datasets in ./data.

Alternatively, we provide the processed [data](https://beyondgeneric.s3.amazonaws.com/data.zip) that can be put into . /data directory. Note that the file_path in each dataset needs to be modified according to the path of the downloaded image in step1. Similarly, some of the parameters in config need to be modified depending on your own.

#### Step3:
Prepare the ckpts of VLP models (take OFA as an example) for training and testing.
1. Download the transformers version ckpts of [OFA](https://huggingface.co/OFA-Sys/ofa-large)
2. However, since there are some [problems](https://github.com/OFA-Sys/OFA/issues/296) with the official ckpts in transformers, we manually replaced the original parameters with the official ckpts in fairseq using `convert_ofa.py`

Alternatively, we provide the converted [ckpts](https://beyondgeneric.s3.amazonaws.com/OFA-large-caption-XEfinetuned.zip).
### Reproduce the main results
***
The baseline result of *OFA* in knowcap: `CUDA_VISIBLE_DEVICES=0 python test_knowcap.py --model OFA --ofa_ckpts xxx --length_penalty 1.0`, the `ofa_ckpts` is obtained in step3.

The *OFA+K-Replay* result in knowcap: `CUDA_VISIBLE_DEVICES=0 python test_knowcap.py --model OFA --trained_ckpts xxx --length_penalty 1.0`, the `trained_ckpts` can be downloaded in [here](https://beyondgeneric.s3.amazonaws.com/model_ofa_kreplay.pt).

To evaluate on coco, use `test.py` instead of `test_knowcap.py`.
### Training with K-Replay
***
#### Step4：
Start Training with K-Replay:
`CUDA_VISIBLE_DEVICES=0 python train_multitask.py --mode train --model OFA --id ofa_kreplay --batch_size 60 --learning_rate 7e-6 --label_smoothing 0.1 --multitask_weight 1.0 --KD_temperature 16.0 --knowdistill_weight 1.0 --save_model_freq 100 --ofa_ckpts /home/chengkz/checkpoints/ofa/OFA-large-caption-trainedenc --ofa_ckpts_distill /home/chengkz/checkpoints/ofa/OFA-large-caption-XEfinetuned --train_mix ./data/train_mix_32000.json --method XEdistill`.

The `ofa_ckpts` and `ofa_ckpts_distill` are obtained in step3, `train_mix_32000.json` is obtained in step2.
#### Step5:
Evaluation on COCO:
`CUDA_VISIBLE_DEVICES=0 python test.py --model OFA --id ofa_kreplay --step 300 --length_penalty 1.0`.

Evaluation on KnowCap: `CUDA_VISIBLE_DEVICES=0 python test_knowcap.py --model OFA --id ofa_kreplay --step 300 --length_penalty 1.0`.
> #### Tips:
OFA uses `resnet` as the backbone of its visual encoder. In our experiments, we found that the `batchnorm` layers in the resnet backbone do not give good estimates of the `mean` and `std` due to the small batchsize we used, which leads to a degradation of the model performance. Therefore, we fixed the `mean` and `std` of these layers during training, by setting `momentum=0.0` in `./transformers/models/ofa/resnet.py`.