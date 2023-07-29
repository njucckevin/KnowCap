import sys
sys.path.append('/home/data_ti4_c/chengkz/ofa/models/OFA')
import torch
import torch.nn as nn
from ofa_model import OFAModel
from transformers.models.ofa.tokenization_ofa import OFATokenizer
import torch.nn.functional as F
from utils.beamsearch import beam_search, beam_search_scst
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OFA(nn.Module):

    def __init__(self, config, distill_model=False):
        super(OFA, self).__init__()
        self.config = config
        self.ofa_ckpts = config.ofa_ckpts
        self.tokenizer = OFATokenizer.from_pretrained(self.ofa_ckpts)
        if distill_model:
            self.ofa_model = OFAModel.from_pretrained(self.config.ofa_ckpts_distill, use_cache=False).to(device)
        else:
            self.ofa_model = OFAModel.from_pretrained(self.config.ofa_ckpts, use_cache=False).to(device)
        #self.ofa_encoder = self.ofa_model.encoder
        self.prompt = " what does the image describe?"
        self.prompt_input = self.tokenizer([self.prompt], return_tensors="pt").input_ids.to(device)
        self.frozen()
        # self.re_init()

    def frozen(self):
        for name, params in self.named_parameters():
            if 'encoder' in name:
                params.requires_grad = False

    def un_frozen(self):
        for name, params in self.named_parameters():
            if 'encoder' in name:
                params.requires_grad = True

    def re_init(self):
        print("reinit decoder")
        self.ofa_model.decoder.init_weights()

    def gen_enc_output(self, patch_img):
        """
        patch_img: [batch_size, *img_patch_size]
        return: [batch_size, 908, 1024]
        """
        batch_size = patch_img.shape[0]
        prompt_input = self.prompt_input.expand([batch_size, self.prompt_input.shape[1]])
        encoder_outputs = self.ofa_model.encoder(input_ids=prompt_input, patch_images=patch_img)
        return encoder_outputs

    def forward(self, patch_img, cap, att_mask, cap_len):
        batch_size = patch_img.shape[0]
        # with torch.no_grad():
        enc_output = self.gen_enc_output(patch_img)
        sentences = cap
        attention_mask = att_mask
        logits = self.ofa_model(decoder_input_ids=sentences,  # [batch_size, cap_len, vocab_size]
                                attention_mask=attention_mask, encoder_outputs=enc_output).logits
        return logits

    def decode_step(self, input_ids, context):
        enc_output = context[0]
        sentences = input_ids
        attention_mask = torch.ones(sentences.shape).long().to(device)
        logits = self.ofa_model(decoder_input_ids=sentences,  # [batch_size, cap_len, vocab_size]
                                attention_mask=attention_mask, encoder_outputs=enc_output).logits
        return logits, None


    def greedy_search(self, patch_img, mode='max'):
        """
        patch_img: [batch_size, *img_patch_size]
        """
        # 贪心搜索，返回的tokens应该是带有开始符和结束符的，以便用作pseudo-caption
        fixed_len = self.config.fixed_len
        gen_num = self.config.beam_num if mode == 'prob' else 1
        batch_size = patch_img.shape[0]*gen_num
        # OFA模型的bos符是0
        sentences = torch.zeros([batch_size, 1]).long().to(device)
        log_probs_sen = torch.full((batch_size, 0), 0.0).to(device)
        cap_len = torch.LongTensor([fixed_len for i in range(batch_size)]).to(device)

        with torch.no_grad():
            enc_output = self.gen_enc_output(patch_img)   # [batch_size, 908, 1024]
        if mode == 'prob':
            enc_output.last_hidden_state = enc_output.last_hidden_state.repeat(1, gen_num, 1). \
                view(enc_output.last_hidden_state.shape[0] * gen_num,
                     enc_output.last_hidden_state.shape[1], enc_output.last_hidden_state.shape[2])
            enc_output.position_embedding = enc_output.position_embedding.repeat(1, gen_num, 1). \
                view(enc_output.position_embedding.shape[0] * gen_num,
                     enc_output.position_embedding.shape[1], enc_output.position_embedding.shape[2])
            enc_output.padding_mask = enc_output.padding_mask.repeat(1, gen_num). \
                view(enc_output.padding_mask.shape[0] * gen_num, enc_output.padding_mask.shape[1])

        for i in range(fixed_len):
            attention_mask = torch.ones(sentences.shape).long().to(device)
            logits_all = self.ofa_model(decoder_input_ids=sentences,     # [batch_size, 1, vocab_size]
                                   attention_mask=attention_mask, encoder_outputs=enc_output).logits
            logits = logits_all[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if mode == 'prob':
                token_id = torch.multinomial(probs, 1)[:, 0]
            else:
                score, token_id = torch.max(probs, dim=-1)
            for j in range(batch_size):  # 生成过程中记录生成句子长度
                if token_id[j].item() == 2 and cap_len[j].item() == fixed_len:
                    cap_len[j] = i + 1
            sentences = torch.cat([sentences, token_id.unsqueeze(1)], dim=1)
            token_id = token_id.unsqueeze(1)
            log_probs_sen = torch.cat([log_probs_sen, torch.log(torch.gather(probs, 1, token_id))], dim=-1)

        all_tokens = [sentences[i][:(cap_len[i] + 1)] for i in range(batch_size)]
        all_logprob = [log_probs_sen[i][:cap_len[i]] for i in range(batch_size)]
        return all_tokens, all_logprob

    def generate_caption_batchbs(self, patch_img):
        batch_size = patch_img.shape[0]
        with torch.no_grad():
            enc_output = self.gen_enc_output(patch_img)
        enc_output.last_hidden_state = enc_output.last_hidden_state.repeat(1, self.config.beam_num, 1).\
            view(enc_output.last_hidden_state.shape[0]*self.config.beam_num, enc_output.last_hidden_state.shape[1], enc_output.last_hidden_state.shape[2])
        enc_output.position_embedding = enc_output.position_embedding.repeat(1, self.config.beam_num, 1).\
            view(enc_output.position_embedding.shape[0]*self.config.beam_num, enc_output.position_embedding.shape[1], enc_output.position_embedding.shape[2])
        enc_output.padding_mask = enc_output.padding_mask.repeat(1, self.config.beam_num).\
            view(enc_output.padding_mask.shape[0]*self.config.beam_num, enc_output.padding_mask.shape[1])
        vocab_size = 59457
        captions = beam_search('Transformer', [enc_output], self, batch_size, self.config.fixed_len, self.config.beam_num,
                              vocab_size, self.config.length_penalty, bos_token_id=0, pad_token_id=1, eos_token_id=2)
        return captions