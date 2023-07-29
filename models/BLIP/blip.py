'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from .vit import VisionTransformer, interpolate_pos_embed
from .med import BertConfig, BertModel, BertLMHeadModel
from utils.beamsearch import beam_search, beam_search_scst
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        

class BLIP_Decoder(nn.Module):
    def __init__(self,
                 config,                 
                 med_config='models/BLIP/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.config = config
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)    
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        
    def forward(self, image, cap, att_mask, cpl = None):
        
        with torch.no_grad():
            image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # 注意，这里的cap是带有prompt的，所以返回的logit也是带有prompt那部分的，目前的代码计算prompt部分的损失
        decoder_output = self.text_decoder(cap, 
                                           attention_mask=att_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,)
        logits = decoder_output.logits
        return logits
        
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        # print(input_ids, input_ids.shape)

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        return outputs
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions


    def decode_step(self, input_ids, context):
        image_embeds = context[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        input_ids[:, 0] = self.tokenizer.bos_token_id
        att_mask = torch.ones(input_ids.shape).long().to(image_embeds.device)
     
        decoder_output = self.text_decoder(input_ids, 
                                           attention_mask=att_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,)
        logits = decoder_output.logits
        return logits, None

    def greedy_search(self, patch_img, mode='max'):
        """
        patch_img: [batch_size, *img_patch_size]
        """
        # 贪心搜索，返回的tokens应该是带有开始符和结束符的，以便用作pseudo-caption
        fixed_len = self.config.fixed_len
        batch_size = patch_img.shape[0]

        with torch.no_grad():
            image_embeds = self.visual_encoder(patch_img)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(patch_img.device)

        # BLIP输入prompt
        prompt = [self.prompt] * patch_img.size(0)
        sentences = self.tokenizer(prompt, return_tensors="pt").input_ids.to(patch_img.device)
        # BLIP forward时的bos token不是berttokenizer的id=101而是自己新设的bos_token_id
        sentences[:, 0] = self.tokenizer.bos_token_id
        sentences = sentences[:, :-1]
        log_probs_sen = torch.full((batch_size, 0), 0.0).to(device)
        cap_len = torch.LongTensor([fixed_len for i in range(batch_size)]).to(device)   # cap_len是在prompt之后生成的长度

        for i in range(fixed_len-self.prompt_length):     # 解码的次数要减去prompt的长度
            att_mask = torch.ones(sentences.shape).long().to(device)
            decoder_output = self.text_decoder(sentences,
                                               attention_mask=att_mask,
                                               encoder_hidden_states=image_embeds,
                                               encoder_attention_mask=image_atts,
                                               return_dict=True, )
            logits_all = decoder_output.logits
            logits = logits_all[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if mode == 'prob':
                token_id = torch.multinomial(probs, 1)[:, 0]
            else:
                score, token_id = torch.max(probs, dim=-1)
            for j in range(batch_size):  # 生成过程中记录生成句子长度
                if token_id[j].item() == self.tokenizer.sep_token_id and cap_len[j].item() == fixed_len:
                    cap_len[j] = i + 1
            sentences = torch.cat([sentences, token_id.unsqueeze(1)], dim=1)
            token_id = token_id.unsqueeze(1)
            log_probs_sen = torch.cat([log_probs_sen, torch.log(torch.gather(probs, 1, token_id))], dim=-1)

        all_tokens = [sentences[i][:(cap_len[i]+1+self.prompt_length-1)] for i in range(batch_size)]
        all_logprob = [log_probs_sen[i][:cap_len[i]] for i in range(batch_size)]
        return all_tokens, all_logprob

    
    def generate_caption_batchbs(self, patch_img):
        batch_size = patch_img.shape[0]
        with torch.no_grad():
            image_embeds = self.visual_encoder(patch_img)
        image_embeds = image_embeds.repeat_interleave(self.config.beam_num, dim=0)

        prompt = [self.prompt] * patch_img.size(0)
        sentences = self.tokenizer(prompt, return_tensors="pt").input_ids.to(patch_img.device)
        sentences[:, 0] = self.tokenizer.bos_token_id
        sentences = sentences[:, :-1]

        vocab_size = 30524
        captions = beam_search('Transformer', [image_embeds], self, batch_size, self.config.fixed_len-self.prompt_length, self.config.beam_num,
                              vocab_size, self.config.length_penalty, bos_token_id=self.tokenizer.bos_token_id, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.sep_token_id, prompt=sentences)
        return captions
    

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # 添加两个，最终词表大小为30524，eos token实际是sep_token[SEP]，id=102
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})    # 添加了[DEC]作为bos_token，自动添加bos_token_id为30522
    # 这个[ENC]是为BLIP的text encoder准备的，在这里不使用
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})  # 添加一个列表的special tokens（但这里只有一个），并自动为每个赋一个id
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
