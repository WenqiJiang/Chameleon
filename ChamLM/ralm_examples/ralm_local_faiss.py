"""
RALM with a local Faiss retriever.

Example usage:
	cd ralm_examples
	# decoder-only
	python ralm_local_faiss.py --batch_size 32 --dim 128 --layers_encoder 2 --layers_decoder 12 --attention_heads 16 --k 10 --retrieval_interval 1 --retrieval_token_len 64 --seq_len 512 --n_batch 1 --model_type decoder
	# encoder-decoder
	python ralm_local_faiss.py --batch_size 32 --dim 128 --layers_encoder 2 --layers_decoder 12 --attention_heads 16 --k 10 --retrieval_interval 1 --retrieval_token_len 64 --seq_len 512 --n_batch 1 --model_type encoder-decoder

"""

import torch
import argparse

from ralm.ralm.ralm import ralmDecoder, ralmEncoderDecoder
from ralm.lm.get_model import createTransformerDecoder, createTransformerEncoderDecoder
from ralm.retriever.faiss_retriever import LocalFaissRetriever

import argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--layers', type=int, default=12)
parser.add_argument('--layers_encoder', type=int, default=2, help="by default use a shallow encoder")
parser.add_argument('--layers_decoder', type=int, default=12)
parser.add_argument('--attention_heads', type=int, default=16)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--retrieval_interval', type=int, default=1)
parser.add_argument('--retrieval_token_len', type=int, default=64, help="length of the retrieval token (only used in encoder-decoder)")
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--n_batch', type=int, default=1, help="number of batches of inference to run")
parser.add_argument('--model_type', type=str, default="decoder", help="decoder or encoder-decoder")


args = parser.parse_args()

batch_size = args.batch_size
dim = args.dim
layers_encoder = args.layers_encoder
layers_decoder = args.layers_decoder
attention_heads = args.attention_heads
k = args.k
retrieval_interval = args.retrieval_interval
retrieval_token_len = args.retrieval_token_len
seq_len = args.seq_len
n_batch = args.n_batch
model_type = args.model_type
assert model_type in ["decoder", "encoder-decoder"]

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

assert dim == 128
# 128-dim SIFT 
dbname = "SIFT1M"
index_key = "IVF4096,PQ16"
index_dir = f"/mnt/scratch/wenqi/Faiss_experiments/trained_CPU_indexes/bench_cpu_{dbname}_{index_key}"
k = 10
nprobe = 32
device = 'gpu'
ngpu = 1
retriever = LocalFaissRetriever(index_dir, dbname, index_key, default_k=k, nprobe=nprobe, device=device, ngpu=ngpu)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Executing on {device}")

if model_type == "encoder-decoder":
    print("Creating encoder-decoder model...")
    model_encoder, model_decoder = createTransformerEncoderDecoder(args=args_model)
    ralm = ralmEncoderDecoder(model_encoder, model_decoder, retriever, 
        batch_size=batch_size, retrieval_interval=retrieval_interval, retrieval_token_len=retrieval_token_len,
        k=k, device=device)
else:
    print("Creating decoder model...")
    model = createTransformerDecoder(args=args_model)
    ralm = ralmDecoder(model, retriever, batch_size=batch_size, retrieval_interval=retrieval_interval, device=device)

# functionality test
# ralm.single_step()
# ralm.multi_steps(num_step=seq_len - 1)

# batch processing
print("Start batch inference...")
for i in range(n_batch):
    print(f"Batch {i}")
    ralm.batch_inference(num_step=seq_len)

ralm.get_profiling()
ralm.print_profiling_stats()