import torch 
import time

from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.data import Dictionary
from fairseq.models.transformer import TransformerConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.decoder = { \
    "embed_dim" : 1024, 
    "ffn_embed_dim": 4096, 
    "layers" : 12,
    "attention_heads" : 16}

cfg = TransformerConfig.from_namespace(args)
print(f"The config created from args: {cfg}")

# vocab to vocab ID
dictionary = Dictionary()

# Input embeddings
vocab_size = 10
dec_embs = torch.nn.Embedding(vocab_size, args.decoder["embed_dim"], dictionary.pad())

batch_size = 64
seq_len = 512

model = TransformerDecoder(args, dictionary, dec_embs, no_encoder_attn=True)
model.to(device)
print(model)

# decoder does not support multi-GPU
# ngpu = 4
# if ngpu > torch.cuda.device_count():
# 	ngpu = torch.cuda.device_count()
# print("Number of GPUs: ", ngpu)

# device_ids (list of python:int or torch.device) – CUDA devices (default: all devices)
# output_device (int or torch.device) – device location of output (default: device_ids[0])
# device_ids = [i for i in range(ngpu)]
# model = torch.nn.DataParallel(model, device_ids=device_ids)
# model.to(device)

# instantiate a new dict for the first time
# out_tensor, out_dict = model(input_tokens)
# incremental_state = dict()
# out_tensor, out_dict = model(input_tokens, incremental_state=incremental_state)

def test_incremental_inference(model, prefix_len=500, final_len=512, batch_size=1, incremental=True):
    """
    Test incremental inference performance. 
    """
    print("Enable incremental inference: ",  incremental)
    if incremental:
        # everytime we just need to consider the latest token and use the previous states
        input_tokens = torch.tensor([[0] * 1] * batch_size).to(device)
        # input_tokens = torch.tensor([[0] * 1] * batch_size)

    # warm up inference
    model(torch.tensor([[0] * 1] * batch_size).to(device))

    with torch.no_grad():

        incremental_state = dict() # initiate
        time_array = []

        for seq_len in range(prefix_len, final_len + 1):
            # input_tokens = torch.tensor([[0] * seq_len] * batch_size).to(device)
            start = time.time()
            """
            The incremental_state will only add the current state into the dictionary, 
                i.e., its size grows linearly with its step steps. 
            Thus, it incremental inference must start from the beginning, i.e.,
                not with a prefix of 500

            Example:
                step 510: 32.01556205749512 ms
                Shape of one layer in incremental_state:  torch.Size([1, 16, 510, 64])
                step 511: 31.69989585876465 ms
                Shape of one layer in incremental_state:  torch.Size([1, 16, 511, 64])
                step 512: 30.326128005981445 ms
                Shape of one layer in incremental_state:  torch.Size([1, 16, 512, 64])
            """
            if incremental:
                out_tensor, out_dict = model(input_tokens, incremental_state=incremental_state)
            else:
                # input_tokens = torch.tensor([[0] * seq_len] * batch_size)
                input_tokens = torch.tensor([[0] * seq_len] * batch_size).to(device)
                out_tensor, out_dict = model(input_tokens)
            end = time.time()
            time_array.append(end - start)
            print('step {}: {} ms'.format(seq_len, (end - start) * 1000))
            if incremental:
                for k in incremental_state:
                    print("Shape of one layer in incremental_state: ", incremental_state[k]['prev_key'].shape)
                    break
                  
test_incremental_inference(model, prefix_len=500, final_len=seq_len, batch_size=batch_size, incremental=False)
test_incremental_inference(model, prefix_len=1, final_len=seq_len, batch_size=batch_size, incremental=True)

# print('output', out_tensor, out_dict)
# print('time consumption: {} ms ({} us per step)'.format((end - start) * 1000, (end - start) * 1e6 / seq_len))


"""
=== Default Model ===

  (dropout_module): FairseqDropout()
  (embed_tokens): Embedding(10, 128, padding_idx=1)
  (project_in_dim): Linear(in_features=128, out_features=512, bias=False)
  (embed_positions): SinusoidalPositionalEmbedding()
  (layers): ModuleList(
    (0-5): 6 x TransformerDecoderLayerBase(
      (dropout_module): FairseqDropout()
      (self_attn): MultiheadAttention(
        (dropout_module): FairseqDropout()
        (k_proj): Linear(in_features=512, out_features=512, bias=True)
        (v_proj): Linear(in_features=512, out_features=512, bias=True)
        (q_proj): Linear(in_features=512, out_features=512, bias=True)
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (activation_dropout_module): FairseqDropout()
      (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (encoder_attn): MultiheadAttention(
        (dropout_module): FairseqDropout()
        (k_proj): Linear(in_features=512, out_features=512, bias=True)
        (v_proj): Linear(in_features=512, out_features=512, bias=True)
        (q_proj): Linear(in_features=512, out_features=512, bias=True)
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (fc1): Linear(in_features=512, out_features=2048, bias=True)
      (fc2): Linear(in_features=2048, out_features=512, bias=True)
      (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
  )
  (output_projection): Linear(in_features=512, out_features=4, bias=False)
)

=== Decoder output (incremental inference) ===

out_tensor: [Batch_size, 1, Vocab_size]
out_tensor
tensor([[[-0.5234,  0.7490,  0.7560,  0.4917]],
        [[-0.3204,  0.2042,  0.3755, -1.8367]]], device='cuda:0')
torch.Size([2, 1, 4])

out_dict has two entries: {'attn': [None], 'inner_states': [tensor([...}

len(out_dict['inner_states'])
13 -> Layer_num + 1 (input embeddings)

for item in out_dict['inner_states']:
    print(item)
    
tensor([[[ 20.4039,  -0.0000, -23.3466,  ..., -40.8960,  38.7881,  -1.5737],
         [ 20.4039, -12.2607, -23.3466,  ...,  -0.0000,  38.7881,  -1.5737]]],
       device='cuda:0')
tensor([[[-0.5845,  0.2275, -0.5924,  ..., -0.7898,  2.0795, -0.2049],
         [ 0.3735, -0.2660, -0.7791,  ...,  0.0776,  1.6020, -0.1366]]],
       device='cuda:0')
       ...
tensor([[[-0.5065, -0.6339,  3.8503,  ..., -0.0151, -0.1864,  0.9715],
         [-0.8171, -0.2381,  2.5466,  ...,  0.9052, -0.2968,  1.0881]]],
       device='cuda:0')
All in shape: torch.Size([1, 2, 1024]) -> [1, Batch_size, Hidden_dim]

Since the first tensor in out_dict['inner_states'] has a quite high variance, while all other
    layers have smaller variance. It could be that the last tensor out_dict['inner_states'][-1] is the last layer state

        
=== Decoder output (default inference) ===

out_tensor.shape
torch.Size([2, 500, 4]) -> (Batch_size, Seq_len, Vocab_size)
out_dict has two entries: {'attn': [None], 'inner_states': [tensor([...}

len(out_dict['inner_states'])
13 -> Layer_num + 1 (input embeddings)

out_dict['inner_states'][-1].shape
torch.Size([500, 2, 1024]) -> (Seq_len, Batch_size, Vocab_size)
"""
