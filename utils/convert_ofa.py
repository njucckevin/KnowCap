# convert the official fairseq version ckpts to the transformers version ckpts
# notice that our K-Replay train the OFA begin with a ckpts with fine-tuned encoder+pre-trained decoder
# eg:
# 1.download official transformers version ckpts in https://huggingface.co/OFA-Sys/ofa-large
# 2.download official fairseq version ckpts in https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_large.pt
# 3.using the following code to obtain the correct transformers version ckpts
import torch
"""
model_t = torch.load('/home/chengkz/checkpoints/ofa/OFA-large/pytorch_model.bin')
model_f = torch.load('/home/chengkz/checkpoints/ofa/OFA-large-fairseq/ofa_large.pt')['model']

key_t = set([k for k in model_t.keys()])
key_f = set([k for k in model_f.keys()])
print(len(key_t), len(key_f))
common_key = key_t.intersection(key_f)
print(len(common_key))

for k in model_t.keys():
  # if 'encoder' in k:
    if k in common_key:
      model_t[k] = model_f[k]
      del model_f[k]
      key_t.remove(k)
      key_f.remove(k)
print(len(key_t), len(key_f))

for k in model_f.keys():
  #if 'encoder' in k:
    k_pred = k.replace('ffn_layernorm', 'ffn_layer_norm')
    k_pred = k_pred.replace('self_attn_ln', 'self_attn_mid_layer_norm')
    k_pred = k_pred.replace('cross_attn_ln', 'cross_attn_mid_layer_norm')
    k_pred = k_pred.replace('encoder_attn', 'cross_attn')
    k_pred = k_pred.replace('attn_ln', 'self_attn_mid_layer_norm')
    if k_pred in key_t:
      model_t[k_pred] = model_f[k]
      key_t.remove(k_pred)
      key_f.remove(k)
print(len(key_t), len(key_f))
print(key_f)

torch.save(model_t, '/home/chengkz/checkpoints/ofa/OFA-large-caption-trainedenc/pytorch_model.bin')
"""

"""
code for BLIP
model_pretrain = torch.load('/home/chengkz/.cache/torch/hub/checkpoints/model_large.pth')
model_ft = torch.load('/home/chengkz/.cache/torch/hub/checkpoints/model_large_caption.pth')['model']
key_ft = set([k for k in model_ft.keys()])
key_ft_vision = {item for item in key_ft if 'visual_encoder' in item}
for k in key_ft_vision:
  model_pretrain['model'][k] = model_ft[k]

torch.save(model_pretrain, '/home/chengkz/.cache/torch/hub/checkpoints/model_large_trainedenc.pth')
"""