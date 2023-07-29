# 基于Transformer架构的图像描述模型
# 包含使用faster-rcnn特征作为输入和cnn特征作为输入

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import pickle
import math
from utils.beamsearch import beam_search, beam_search_scst
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_Encoder(nn.Module):

    def __init__(self, config):
        super(Transformer_Encoder, self).__init__()
        self.config = config
        self.image_dim = config.image_dim
        self.embed_dim = config.embed_dim
        self.fea2embed = nn.Linear(self.image_dim, self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, fea_maps):
        fea_maps = self.fea2embed(fea_maps)
        fea_maps_seq = fea_maps.permute(1, 0, 2)
        memory = self.transformer_encoder(src=fea_maps_seq)
        return memory


class Transformer_Decoder(nn.Module):

    def __init__(self, config):
        super(Transformer_Decoder, self).__init__()
        self.config = config
        self.vocab = pickle.load(open(self.config.vocab, 'rb'))
        self.vocab_size = self.vocab.get_size()
        self.embed_dim = config.embed_dim

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)

        self.pos_encoder = PositionalEncoding(self.embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.fc = weight_norm(nn.Linear(self.embed_dim, self.vocab_size))
        self.dropout = nn.Dropout(0.5)

    def gen_tgt_mask(self, length):
        mask = torch.triu(torch.ones(length, length)).permute(1, 0).to(device)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask

    def forward(self, memory, cap, cap_len):
        cap = cap.permute(1, 0)
        tgt_pos_embedding = self.pos_encoder(self.embed(cap)*math.sqrt(self.embed_dim))
        tgt_mask = self.gen_tgt_mask(tgt_pos_embedding.shape[0])
        out = self.transformer_decoder(tgt=tgt_pos_embedding, memory=memory, tgt_mask=tgt_mask)

        pred = self.fc(self.dropout(out))
        pred = pred.permute(1, 0, 2)

        return pred

    def decode_step(self, input_ids, context):
        memory = context[0]
        cap = input_ids.permute(1, 0)
        tgt_pos_embedding = self.pos_encoder(self.embed(cap) * math.sqrt(self.embed_dim))
        tgt_mask = self.gen_tgt_mask(tgt_pos_embedding.shape[0])
        out = self.transformer_decoder(tgt=tgt_pos_embedding, memory=memory, tgt_mask=tgt_mask)

        pred = self.fc(self.dropout(out))
        pred = pred.permute(1, 0, 2)
        return pred, None


class Transformer_Cap(nn.Module):

    def __init__(self, config):
        super(Transformer_Cap, self).__init__()
        self.config = config
        self.transformer_encoder = Transformer_Encoder(self.config)
        self.transformer_decoder = Transformer_Decoder(self.config)

    def forward(self, image_feature, cap, cap_len, mode='xe'):
        if mode == 'xe':
            fea_maps = image_feature['feature_map']
            memory = self.transformer_encoder(fea_maps)
            logit = self.transformer_decoder(memory, cap, cap_len)
            return logit
        elif mode == 'vanilla_scst':
            return self.greedy_search(image_feature, 'prob')

    def beam_search(self, image_feature):
        fea_maps = image_feature['feature_map']
        batch_size = fea_maps.shape[0]
        memory = self.transformer_encoder(fea_maps)
        memory = memory.repeat(1, 1, self.config.beam_num).view(memory.shape[0], memory.shape[1]*self.config.beam_num, memory.shape[2])
        captions, all_tokens, all_logprob = beam_search_scst('Transformer', [memory], self.transformer_decoder, batch_size, self.config.fixed_len, self.config.beam_num,
                              self.transformer_decoder.vocab_size, self.config.length_penalty)
        return captions, all_tokens, all_logprob

    def greedy_search(self, image_feature, mode='max'):
        # greedy search或多项式采样search
        fea_maps = image_feature['feature_map']
        # 对一个样本采样beam_num个结果
        gen_num = self.config.beam_num if mode == 'prob' else 1
        fea_maps = fea_maps.unsqueeze(dim=1)
        fea_maps = fea_maps.expand([fea_maps.shape[0], gen_num, fea_maps.shape[2], fea_maps.shape[3]])
        fea_maps = fea_maps.reshape(fea_maps.shape[0] * fea_maps.shape[1], fea_maps.shape[2], fea_maps.shape[3])
        batch_size = fea_maps.shape[0]

        sentences = torch.ones([batch_size, 1]).to(device).long()
        log_probs_sen = torch.full((batch_size, 0), 0.0).to(device)
        cap_len = torch.LongTensor([20 for i in range(batch_size)]).to(device)

        memory = self.transformer_encoder(fea_maps)
        context = [memory]
        for i in range(self.config.fixed_len):
            outputs, _ = self.transformer_decoder.decode_step(sentences, context)
            logits = outputs[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if mode == 'prob':
                token_id = torch.multinomial(probs, 1)[:, 0]
            else:
                score, token_id = torch.max(probs, dim=-1)
            for j in range(batch_size):  # 生成过程中记录生成句子长度
                if token_id[j].item() == 2 and cap_len[j].item() == 20:
                    cap_len[j] = i + 1
            sentences = torch.cat([sentences, token_id.unsqueeze(1)], dim=1)
            token_id = token_id.unsqueeze(1)
            log_probs_sen = torch.cat([log_probs_sen, torch.log(torch.gather(probs, 1, token_id))], dim=-1)

        # 利用生成句子长度mask
        all_tokens = [sentences[i][:(cap_len[i] + 1)] for i in range(batch_size)]
        all_logprob = [log_probs_sen[i][:cap_len[i]] for i in range(batch_size)]

        return all_tokens, all_logprob

    def generate_caption_batchbs(self, image_feature):
        fea_maps = image_feature['feature_map']
        batch_size = fea_maps.shape[0]
        memory = self.transformer_encoder(fea_maps)
        memory = memory.repeat(1, 1, self.config.beam_num).view(memory.shape[0], memory.shape[1]*self.config.beam_num, memory.shape[2])
        caption = beam_search('Transformer', [memory], self.transformer_decoder, batch_size, self.config.fixed_len, self.config.beam_num,
                              self.transformer_decoder.vocab_size, self.config.length_penalty)
        return caption

