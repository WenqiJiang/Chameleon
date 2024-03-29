import torch 
import time

# Data parallel: 
# https://pytorch.org/docs/1.7.0/generated/torch.nn.DataParallel.html?highlight=data%20parallel#torch.nn.DataParallel

"""
Some performance number (up to 4 x 3090):

	1 GPU max batch size = 32

	For encoder, the GPU throughput is close to optimal, even without batching:
		batch size: 1, ngpu: 1, seq_len: 512, time: 16.03 ms, inference / sec: 62.37
		batch size: 2, ngpu: 1, seq_len: 512, time: 24.96 ms, inference / sec: 80.13
		batch size: 4, ngpu: 1, seq_len: 512, time: 45.50 ms, inference / sec: 87.91
		batch size: 4, ngpu: 1, seq_len: 512, time: 85.29 ms, inference / sec: 93.80
		batch size: 8, ngpu: 1, seq_len: 512, time: 85.29 ms, inference / sec: 93.80
		batch size: 16, ngpu: 1, seq_len: 512, time: 167.84 ms, inference / sec: 95.33
		batch size: 32, ngpu: 1, seq_len: 512, time: 325.73 ms, inference / sec: 98.24
		
	For multi-GPU, 4 GPU can achieve ~3x speedup over a single GPU:
		batch size: 64, ngpu: 2, seq_len: 512, time: 359.49 ms, inference / sec: 178.03
		batch size: 64, ngpu: 4, seq_len: 512, time: 228.46 ms, inference / sec: 280.14
		
		batch size: 128, ngpu: 4, seq_len: 512, time: 437.81 ms, inference / sec: 292.36
"""

from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.data import Dictionary
from fairseq.models.transformer import TransformerConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.encoder = { \
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
enc_embs = torch.nn.Embedding(vocab_size, args.encoder["embed_dim"], dictionary.pad())

model = TransformerEncoder(args, dictionary, enc_embs)
print(model)

ngpu = 4
if ngpu > torch.cuda.device_count():
	ngpu = torch.cuda.device_count()
print("Number of GPUs: ", ngpu)
batch_size = 32
seq_len = 512
input_tokens = torch.tensor([[1] * seq_len] * batch_size).to(device)

# device_ids (list of python:int or torch.device) – CUDA devices (default: all devices)
# output_device (int or torch.device) – device location of output (default: device_ids[0])
device_ids = [i for i in range(ngpu)]
model = torch.nn.DataParallel(model, device_ids=device_ids)
# model = torch.nn.DataParallel(model, device_ids=[0])
model.to(device)
# time.sleep(10)

"""
encoder output dict:
    - **encoder_out** (Tensor): the last encoder layer's output of
        shape `(src_len, batch, embed_dim)`
    - **encoder_padding_mask** (ByteTensor): the positions of
        padding elements of shape `(batch, src_len)`
    - **encoder_embedding** (Tensor): the (scaled) embedding lookup
        of shape `(batch, src_len, embed_dim)`
    - **encoder_states** (List[Tensor]): all intermediate
        hidden states of shape `(src_len, batch, embed_dim)`.
        Only populated if *return_all_hiddens* is True.
"""

out_dict = model(input_tokens)

with torch.no_grad():
    start = time.time()
    out_dict = model(input_tokens)
    end = time.time()
                  
print('\noutput:\n', out_dict)
print('\nencoder_out:\n', out_dict['encoder_out'], out_dict['encoder_out'][0].shape)
print('\nencoder_padding_mask:\n', out_dict['encoder_padding_mask'], out_dict['encoder_padding_mask'][0].shape)
print('\nencoder_embedding:\n', out_dict['encoder_embedding'], out_dict['encoder_embedding'][0].shape)
print('\nencoder_states:\n', out_dict['encoder_states'])
print('time consumption: {:.2f} ms'.format((end - start) * 1000))
print('inference / sec: {:.2f}'.format(batch_size / (end - start)))
"""
out_dict
{'encoder_out': [tensor([[[nan, nan, ...='cuda:0')], 'encoder_padding_mask': [tensor([[True, True,...='cuda:0')], 'encoder_embedding': [tensor([[[0., 0., 0....='cuda:0')], 'encoder_states': [], 'fc_results': [], 'src_tokens': [], 'src_lengths': [tensor([[0],
       ...rch.int32)]}
       
out_dict['encoder_out']
[tensor([[[nan, nan, ...='cuda:0')]

out_dict['encoder_out'][0].shape
torch.Size([512, 2, 1024]) -> [Seq_len, Batch_size, Hidden_dim]

out_dict['encoder_embedding']
[tensor([[[0., 0., 0....='cuda:0')]

out_dict['encoder_embedding'][0].shape
torch.Size([2, 512, 1024]) -> [Batch_size, Seq_len, Hidden_dim]
"""