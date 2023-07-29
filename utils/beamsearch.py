# batch beamsearch
# 参照huggingface的实现 https://zhuanlan.zhihu.com/p/167072494 http://www.wuyuanhao.com/2020/03/20/解读beam-search-1-2/
# 除了支持以batch形式一次为多个样本进行beamsearch，与传统beamsearch的最大不同在于：
# 对于beam中的序列，即使生成了end标识符，beam的宽度也不会减小；而是将生成完成的序列存入BeamHypotheses，并向beam中补充一个新的未生成完成序列，
# 并继续宽度为beam的搜索过程，期间不断用新生成完成的序列更新BeamHypotheses

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BeamHypotheses(object):
    # 每个样本绑定一个，其中维护num_beams个当前最优的序列；可向其中添加新序列并自动踢掉分数最低的
    def __init__(self, num_beams, max_length, length_penalty):
        # 初始化
        self.max_length = max_length - 1
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        # 长度惩罚，可自定义
        score = sum_logprobs / len(hyp) ** self.length_penalty
        # score = sum_logprobs / (pow((5+len(hyp)+1), self.length_penalty)/pow(5+1, self.length_penalty))
        if len(self) < self.num_beams or score > self.worst_score:
            # 可添加
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # 需要删掉一个
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def add_scst(self, hyp, logprob, sum_logprobs):
        # 长度惩罚，可自定义
        score = sum_logprobs / len(hyp) ** self.length_penalty
        # score = sum_logprobs / (pow((5+len(hyp)+1), self.length_penalty)/pow(5+1, self.length_penalty))
        if len(self) < self.num_beams or score > self.worst_score:
            # 可添加
            self.beams.append((score, hyp, logprob))
            if len(self) > self.num_beams:
                # 需要删掉一个
                sorted_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        # 样本是否已经生成完成，关键：并非生成beam个完成的序列，而是新一时刻beam宽度个结果中的最高分不如之前保存的最低分
        # best_sum_logprobs是新的候选序列中的最高得分
        if len(self) < self.num_beams:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # cur_score = best_sum_logprobs / (pow((5+cur_len+1), self.length_penalty)/pow(5+1, self.length_penalty))
            # 如果最高分比保存的最低分还差，则结束
            ret = self.worst_score >= cur_score
            return ret


def beam_search(mode, context, model, batch_size, max_length, num_beams, vocab_size, length_penalty,
                bos_token_id=1, pad_token_id=0, eos_token_id=2, prompt=None):
    # batch beamsearch
    # 记录每个样本的已生成序列，已生成序列得分和是否已生成完成
    generated_hyps = [BeamHypotheses(num_beams, max_length, length_penalty) for _ in range(batch_size)]
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float).to(device)
    beam_scores[:, 1:] = -1e9   # 否则t=1时刻取到的num_beams个最大的将都是同一个词，从而导致后面所有num_beams个结果均相同
    beam_scores = beam_scores.view(-1)
    done = [False for _ in range(batch_size)]

    # 初始input和当前长度
    if prompt == None:
        input_ids = torch.full((batch_size*num_beams, 1), bos_token_id, dtype=torch.long).to(device)
    else:
        input_ids = prompt.repeat([num_beams, 1])
    cur_len = 1

    # 初始状态 hidden: (batch_size*num_beams, *)
    # 对于LSTM-based模型来说，hidden是解码器的隐藏层状态，需要在每个时刻更新；而对于Transformer-based模型来说，hidden是编码端的输出，解码所有时刻保持不变
    # hidden = context

    while cur_len < max_length:
        # 需要模型实现一个接口：根据hidden状态，以及当前已生成的序列，生成下一时刻的词表概率分布（以及LSTM-based模型需要更新后的hidden）
        outputs, hidden = model.decode_step(input_ids, context)
        next_token_logits = outputs[:, -1, :]

        scores = F.log_softmax(next_token_logits, dim=-1)
        next_scores = scores + beam_scores[:, None].expand_as(scores)
        next_scores = next_scores.view(batch_size, num_beams*vocab_size)    # 便于用topk为batch内的每个样本选最大

        # next_scores/next_tokens: (batch_size, num_beams)
        # 关键：这里保留了2*num_beams个结果，目的是即使有beam生成了eos，依然能找到num_beams可以继续生成的选项
        next_scores, next_tokens = torch.topk(next_scores, 2*num_beams, dim=1, largest=True, sorted=True)

        next_batch_beam = []    # 为下一时刻准备 (分数, token_id, beam_id)
        for batch_idx in range(batch_size):
            if done[batch_idx]:     # 如果当前batch已经完成，直接补pad
                next_batch_beam.extend([(0, pad_token_id, 0)]*num_beams)
                continue
            next_sent_beam = []     # 记录一个batch内beam_num个最好的（且没有生成完成的）结果
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                beam_id = beam_token_id // vocab_size   # beam_id：属于当前batch的第几个beam
                token_id = beam_token_id % vocab_size
                effective_beam_id = batch_idx * num_beams + beam_id     # 在原始(batch_size*num_beams, *)中的位置
                if token_id.item() == eos_token_id:
                    # 生成eos，将当前beam的句子存入
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    # 存入时不包含eos
                    generated_hyps[batch_idx].add(input_ids[effective_beam_id].clone(), beam_token_score.item())
                else:
                    # 保存生成后的状态
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                if len(next_sent_beam) == num_beams:    # 当前batch不管有没有、生成了几个eos，依然会保留num_beams个可扩展的序列
                    break

            # 什么情况算生成完成？已经生成了num_beams个完整句子，且当前时刻生成的结果（可能是完整句子，也可能不是）没有新的更好的
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx]\
                .is_done(next_scores[batch_idx].max().item(), cur_len)

            next_batch_beam.extend(next_sent_beam)

        if all(done):
            break

        # 准备下一时刻
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)

        if mode == 'LSTM':  # LSTM需要更新隐藏层状态
            hidden = [item[beam_idx, :] for item in hidden]
            #h, c = hidden
            #h = h[beam_idx, :]
            #c = c[beam_idx, :]
            #hidden = (h, c)
            context[-1] = hidden

        cur_len += 1

    # 手动结束没有生成eos的样本
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        for beam_id in range(num_beams):
            # 对于需要手动结束的样本，全部尝试加入
            effective_beam_id = batch_idx*num_beams+beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # 至此，generated_hyps中保存着每个样本的num_beams个最优序列
    best = []
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        best_hyp = sorted_hyps.pop()[1]
        best.append(best_hyp)

    return best


def beam_search_scst(mode, context, model, batch_size, max_length, num_beams, vocab_size, length_penalty,
                bos_token_id=1, pad_token_id=0, eos_token_id=2):
    # batch beamsearch
    # 记录每个样本的已生成序列，已生成序列得分和是否已生成完成
    # 在beamseach的每个时刻，保存当前最优beam个从开始到当前所有时刻的logprob
    generated_hyps = [BeamHypotheses(num_beams, max_length, length_penalty) for _ in range(batch_size)]
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float).to(device)
    beam_scores[:, 1:] = -1e9   # 否则t=1时刻取到的num_beams个最大的将都是同一个词，从而导致后面所有num_beams个结果均相同
    beam_scores = beam_scores.view(-1)
    done = [False for _ in range(batch_size)]

    # 初始input和当前长度
    input_ids = torch.full((batch_size*num_beams, 1), bos_token_id, dtype=torch.long).to(device)
    ids_logprob = torch.full((batch_size*num_beams, 0), 0.0).to(device)
    cur_len = 1

    # 初始状态 hidden: (batch_size*num_beams, *)
    # 对于LSTM-based模型来说，hidden是解码器的隐藏层状态，需要在每个时刻更新；而对于Transformer-based模型来说，hidden是编码端的输出，解码所有时刻保持不变
    # hidden = context

    while cur_len < max_length:
        # 需要模型实现一个接口：根据hidden状态，以及当前已生成的序列，生成下一时刻的词表概率分布（以及LSTM-based模型需要更新后的hidden）
        outputs, hidden = model.decode_step(input_ids, context)
        next_token_logits = outputs[:, -1, :]
        scores = F.log_softmax(next_token_logits, dim=-1)
        next_scores = scores + beam_scores[:, None].expand_as(scores)
        next_scores = next_scores.view(batch_size, num_beams*vocab_size)    # 便于用topk为batch内的每个样本选最大
        scores = scores.view(batch_size, num_beams*vocab_size)  # 便于根据取出topk的id取出对应的概率

        # next_scores/next_tokens: (batch_size, num_beams)
        # 关键：这里保留了2*num_beams个结果，目的是即使有beam生成了eos，依然能找到num_beams可以继续生成的选项
        next_scores, next_tokens = torch.topk(next_scores, 2*num_beams, dim=1, largest=True, sorted=True)

        next_batch_beam = []    # 为下一时刻准备 (分数, token_id, beam_id)
        for batch_idx in range(batch_size):
            if done[batch_idx]:     # 如果当前batch已经完成，直接补pad
                next_batch_beam.extend([(0, pad_token_id, 0, 0)]*num_beams)
                continue
            next_sent_beam = []     # 记录一个batch内beam_num个最好的（且没有生成完成的）结果
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                beam_id = beam_token_id // vocab_size   # beam_id：属于当前batch的第几个beam
                token_id = beam_token_id % vocab_size
                logprob = scores[batch_idx][beam_token_id]
                effective_beam_id = batch_idx * num_beams + beam_id     # 在原始(batch_size*num_beams, *)中的位置
                if token_id.item() == eos_token_id:
                    # 生成eos，将当前beam的句子存入
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    # 存入时不包含eos
                    logprob_add = torch.cat([ids_logprob[effective_beam_id].clone(), logprob.unsqueeze(0)], dim=0)
                    generated_hyps[batch_idx].add_scst(input_ids[effective_beam_id].clone(), logprob_add, beam_token_score.item())

                else:
                    # 保存生成后的状态
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id, logprob))

                if len(next_sent_beam) == num_beams:    # 当前batch不管有没有、生成了几个eos，依然会保留num_beams个可扩展的序列
                    break

            # 什么情况算生成完成？已经生成了num_beams个完整句子，且当前时刻生成的结果（可能是完整句子，也可能不是）没有新的更好的
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx]\
                .is_done(next_scores[batch_idx].max().item(), cur_len)

            next_batch_beam.extend(next_sent_beam)

        if all(done):
            break

        # 准备下一时刻
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])
        beam_logprob = ids_logprob.new([x[3] for x in next_batch_beam])

        input_ids = input_ids[beam_idx, :]
        ids_logprob = ids_logprob[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        ids_logprob = torch.cat([ids_logprob, beam_logprob.unsqueeze(1)], dim=-1)

        if mode == 'LSTM':  # LSTM需要更新隐藏层状态
            hidden = [item[beam_idx, :] for item in hidden]
            #h, c = hidden
            #h = h[beam_idx, :]
            #c = c[beam_idx, :]
            #hidden = (h, c)
            context[-1] = hidden

        cur_len += 1

    # 手动结束没有生成eos的样本
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        for beam_id in range(num_beams):
            # 对于需要手动结束的样本，全部尝试加入
            effective_beam_id = batch_idx*num_beams+beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            final_logprob = ids_logprob[effective_beam_id]
            generated_hyps[batch_idx].add_scst(final_tokens, final_logprob, final_score)

    # 至此，generated_hyps中保存着每个样本的num_beams个最优序列
    best = []
    all_tokens = []
    all_logprob = []
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        best_hyp = sorted_hyps[-1][1]
        best.append(best_hyp)
        all_tokens.extend([item[1] for item in sorted_hyps])
        all_logprob.extend([item[2] for item in sorted_hyps])

    return best, all_tokens, all_logprob

