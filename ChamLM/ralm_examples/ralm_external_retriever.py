"""
Resort to external retriever, and connect to a server for RALM inference

Example Usage (single GPU):

  In one terminal, start the server:
    cd ralm/server
    python server.py --host 127.0.0.1 --port 9091 --batch_size 32 --dim 512 --k 10 --nprobe 32 --request_with_lists 1 --delay_ms 0

  In another terminal, run the inference:
    cd ralm_examples
    python ralm_external_retriever.py --dim 512 --layers_encoder 2 --layers_decoder 12 --attention_heads 16 --model_type decoder \
    --host 127.0.0.1 --port 9091  --k 10 --retrieval_interval 1 --retrieval_token_len 64 --seq_len 512 --request_with_lists 1 --nlist 32768 --nprobe 32 \
    --batch_size 32 --n_batch 1 --use_gpu_id 0  --use_tiktok 1
    
    
Example Usage (multi-GPU):

  In terminal 1, start the server:
    cd ralm/server
	python server.py --host 127.0.0.1 --port 9091 --batch_size 32 --dim 512 --k 10 --nprobe 32 --request_with_lists 1 --delay_ms 0
    
  In terminal 2, start the cooridnator 
  	# NOTE! num_queries_per_gpu must be calculated by ceil(seq_len / retrieval_interval) * batch_size), x2 for tik-tok:
    cd ralm/coordinator
    python retriever_coordinator_server.py --search_server_host 127.0.0.1 --search_server_port 9091 \
        --local_port 9090 --ngpus 2 --batch_size 32 --num_queries_per_gpu 1024 --k 10 --dim 512 --nprobe 32 --request_with_lists 1 
         
  In terminal 3 ~ 4, start the RALM processes:
    cd ralm_examples
    
    # set different gpu IDs
	python ralm_external_retriever.py --dim 512 --layers_encoder 2 --layers_decoder 12 --attention_heads 16 --model_type decoder \
    --host 127.0.0.1 --port 9090  --k 10 --retrieval_interval 1 --retrieval_token_len 64 --seq_len 512 --request_with_lists 1 --nlist 32768 --nprobe 32 \
    --batch_size 32 --n_batch 1 --use_gpu_id 0  --use_tiktok 1 --use_coordinator 1

	python ralm_external_retriever.py --dim 512 --layers_encoder 2 --layers_decoder 12 --attention_heads 16 --model_type decoder \
    --host 127.0.0.1 --port 9090  --k 10 --retrieval_interval 1 --retrieval_token_len 64 --seq_len 512 --request_with_lists 1 --nlist 32768 --nprobe 32 \
    --batch_size 32 --n_batch 1 --use_gpu_id 1  --use_tiktok 1	--use_coordinator 1
"""


import torch

from ralm.ralm.ralm import ralmDecoder, ralmEncoderDecoder

from ralm.ralm.ralm_tiktok import ralmTikTokDecoder, ralmTikTokEncoderDecoder
from ralm.lm.get_model import createTransformerDecoder, createTransformerEncoderDecoder
from ralm.retriever.retriever import ExternalRetriever
from ralm.index_scanner.index_scanner import IndexScanner

import argparse 
parser = argparse.ArgumentParser()

# model params
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--layers_encoder', type=int, default=2, help="by default use a shallow encoder")
parser.add_argument('--layers_decoder', type=int, default=12)
parser.add_argument('--attention_heads', type=int, default=16)
parser.add_argument('--model_type', type=str, default="decoder", help="decoder or encoder-decoder")

# retrieval params
parser.add_argument('--host', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=9090)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--retrieval_interval', type=int, default=1)
parser.add_argument('--retrieval_token_len', type=int, default=64, help="length of the retrieval token (only used in encoder-decoder)")
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--request_with_lists', type=int, default=0, help="whether to scan index locally on the GPU")
parser.add_argument('--nlist', type=int, default=32768, help="number of clusters in the index")
parser.add_argument('--nprobe', type=int, default=32, help="number of clusters to scan")

# runtime params
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_batch', type=int, default=1, help="number of batches of inference to run")
parser.add_argument('--use_gpu_id', type=int, default=None, help="the GPU ID to run the model")
parser.add_argument('--use_tiktok', type=int, default=0, help="0 or 1, whether to use tiktok")
parser.add_argument('--use_coordinator', type=int, default=0, help="0 or 1, whether to use the coordinator")


args = parser.parse_args()

dim = args.dim
layers_encoder = args.layers_encoder
layers_decoder = args.layers_decoder
attention_heads = args.attention_heads
model_type = args.model_type
assert model_type in ["decoder", "encoder-decoder"]

host = args.host
port = args.port
k = args.k
retrieval_interval = args.retrieval_interval
retrieval_token_len = args.retrieval_token_len
seq_len = args.seq_len
request_with_lists = args.request_with_lists
nlist = args.nlist
nprobe = args.nprobe

batch_size = args.batch_size
n_batch = args.n_batch
use_gpu_id = args.use_gpu_id
use_tiktok = args.use_tiktok
assert use_tiktok in [0, 1]
use_coordinator = args.use_coordinator

n_retrieve_per_seq = int(seq_len / retrieval_interval)
if use_tiktok:
    n_retrieve_per_seq *= 2
print("The server needs to answer {} batches of queries".format(int(n_retrieve_per_seq * n_batch)))
print("Please start the server before this process..., e.g., \n"
      "cd ralm/server\n"
      "python server.py")

retriever = ExternalRetriever(default_k=k, host=host, port=port, batch_size=batch_size, dim=dim)

args_model = parser.parse_args()
args_model.decoder = { \
    "embed_dim" : dim, 
    "ffn_embed_dim": dim * 4, 
    "layers" : layers_decoder,
    "attention_heads" : attention_heads}
if model_type == "encoder-decoder":
    args_model.encoder = { \
        "embed_dim" : dim, 
        "ffn_embed_dim": dim * 4, 
        "layers" : layers_encoder,
        "attention_heads" : attention_heads}
    
if torch.cuda.is_available():
    if use_gpu_id is not None:
        device = f'cuda:{use_gpu_id}'
    else:
        device = 'cuda'
else:
    device = 'cpu'
    
if request_with_lists:
    print("Creating GPU index scanner...")
    index_scanner = IndexScanner(dim=dim, nlist=nlist, nprobe=nprobe, use_gpu_id=use_gpu_id)
else:
    index_scanner = None
    
print(f"Executing on {device}")

if use_tiktok:
    print("Using tiktok scheduling...")
else:
    print("Disabled tiktok scheduling...")

if model_type == "encoder-decoder":
    print("Creating encoder-decoder model...")
    model_encoder, model_decoder = createTransformerEncoderDecoder(args=args_model)
    if use_tiktok:
        ralm = ralmTikTokEncoderDecoder(model_encoder, model_decoder, retriever, index_scanner=index_scanner,
            batch_size=batch_size, retrieval_interval=retrieval_interval, retrieval_token_len=retrieval_token_len, 
            k=k, device=device, use_coordinator=use_coordinator)
    else:
        ralm = ralmEncoderDecoder(model_encoder, model_decoder, retriever, index_scanner=index_scanner,
            batch_size=batch_size, retrieval_interval=retrieval_interval, retrieval_token_len=retrieval_token_len, 
            k=k, device=device, use_coordinator=use_coordinator)
else:
    print("Creating decoder model...")
    model_decoder = createTransformerDecoder(args=args_model)
    if use_tiktok:
        ralm = ralmTikTokDecoder(model_decoder, retriever, index_scanner=index_scanner, batch_size=batch_size, 
			retrieval_interval=retrieval_interval, device=device, use_coordinator=use_coordinator)
    else:
        ralm = ralmDecoder(model_decoder, retriever, index_scanner=index_scanner, batch_size=batch_size, 
			retrieval_interval=retrieval_interval, device=device, use_coordinator=use_coordinator)


# batch processing
print("Start batch inference...")
for i in range(n_batch):
    print(f"Batch {i}")
    ralm.batch_inference(num_step=seq_len)

ralm.get_profiling()
ralm.print_profiling_stats()