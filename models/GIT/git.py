import sys
sys.path.append('/home/data_ti4_c/chengkz/ofa/models/GIT')
import torch
import torch.nn as nn
from git_model import GitForCausalLM
from transformers import AutoProcessor
import torch.nn.functional as F
from utils.beamsearch import beam_search, beam_search_scst
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GIT(nn.Module):

    def __init__(self, config, distill_model=False):
        super(GIT, self).__init__()
        self.config = config
        self.processor = AutoProcessor.from_pretrained("microsoft/git-large-coco", local_files_only=False)
        self.tokenizer = self.processor.tokenizer
        if distill_model:
            self.git_model = GitForCausalLM.from_pretrained(config.git_distill)
        else:
            self.git_model = GitForCausalLM.from_pretrained(config.git)

    def get_enc_output(self, patch_img):
        # 为了实现decode_step函数，需要将视觉编码单独分割出来，避免decode_step中每一步都forward视觉编码器
        # 同时也改写了GitModel的forward接口，允许提供视觉编码结果以避免重复计算
        projected_visual_features = None
        if patch_img is not None:
            if patch_img.ndim == 4:
                # here we assume patch_img is of shape (batch_size, num_channels, height, width)
                visual_features = self.git_model.git.image_encoder(patch_img).last_hidden_state
            elif patch_img.ndim == 5:
                # here we assume patch_img is of shape (batch_size, num_frames, num_channels, height, width)
                visual_features = []
                for frame_idx in range(patch_img.shape[1]):
                    visual_features_frame = self.git_model.git.image_encoder(patch_img[:, frame_idx, :, :]).last_hidden_state
                    visual_features_frame += self.git_model.git.img_temperal_embedding[frame_idx]
                    visual_features.append(visual_features_frame)

                # finally, concatenate all features along sequence dimension
                visual_features = torch.cat(visual_features, dim=1)
            else:
                raise ValueError("patch_img must be of rank 4 or 5")
            projected_visual_features = self.git_model.git.visual_projection(visual_features)
        return projected_visual_features

    def forward(self, patch_img, cap, att_mask, cap_len):
        batch_size = patch_img.shape[0]
        with torch.no_grad():
            visual_features = self.get_enc_output(patch_img)
        logits = self.git_model(input_ids=cap, attention_mask=att_mask, visual_features=visual_features, pixel_values=patch_img).logits
        logits = logits[:, -20:, :]
        return logits

    def decode_step(self, input_ids, context):
        visual_features = context[0]
        patch_img = context[1]
        att_mask = torch.ones(input_ids.shape).long().to(device)
        logits = self.git_model(input_ids=input_ids, attention_mask=att_mask, visual_features=visual_features, pixel_values=patch_img).logits
        return logits, None

    def greedy_search(self, patch_img, mode='max'):
        """
        patch_img: [batch_size, *img_patch_size]
        """
        # 贪心搜索，返回的tokens应该是带有开始符和结束符的，以便用作pseudo-caption
        fixed_len = self.config.fixed_len
        gen_num = self.config.beam_num if mode == 'prob' else 1
        batch_size = patch_img.shape[0]*gen_num
        # GIT模型的bos符是bert的cls符，101
        sentences = torch.full((batch_size, 1), self.tokenizer.cls_token_id).long().to(device)
        log_probs_sen = torch.full((batch_size, 0), 0.0).to(device)
        cap_len = torch.LongTensor([fixed_len for i in range(batch_size)]).to(device)

        with torch.no_grad():
            visual_features = self.get_enc_output(patch_img)

        for i in range(fixed_len):
            attention_mask = torch.ones(sentences.shape).long().to(device)
            logits_all = self.git_model(input_ids=sentences, attention_mask=attention_mask, visual_features=visual_features,
                                    pixel_values=patch_img).logits
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

        all_tokens = [sentences[i][:(cap_len[i] + 1)] for i in range(batch_size)]
        all_logprob = [log_probs_sen[i][:cap_len[i]] for i in range(batch_size)]
        return all_tokens, all_logprob

    def generate_caption_batchbs(self, patch_img):
        batch_size = patch_img.shape[0]
        with torch.no_grad():
            visual_features = self.get_enc_output(patch_img)
        visual_features = visual_features.repeat_interleave(self.config.beam_num, dim=0)

        vocab_size = 30522
        captions = beam_search('Transformer', [visual_features, patch_img], self, batch_size, self.config.fixed_len, self.config.beam_num,
                              vocab_size, self.config.length_penalty, bos_token_id=self.tokenizer.cls_token_id, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.sep_token_id)
        return captions












