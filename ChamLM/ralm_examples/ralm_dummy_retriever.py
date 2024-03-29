"""
RALM with a dummy retriever.

Example usage:
    cd ralm_examples
    python ralm_dummy_retriever.py --dim 512 --layers_encoder 2 --layers_decoder 12 --attention_heads 16 --model_type decoder \
        --k 10 --retrieval_interval 1 --retrieval_token_len 64 --seq_len 512 --request_with_lists 1 --nlist 32768 --nprobe 32 \
        --batch_size 32 --n_batch 1 --use_gpu_id 0

"""
import torch

from ralm.ralm.ralm import ralmDecoder, ralmEncoderDecoder
from ralm.lm.get_model import createTransformerDecoder, createTransformerEncoderDecoder
from ralm.retriever.retriever import DummyRetriever
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


args = parser.parse_args()

dim = args.dim
layers_encoder = args.layers_encoder
layers_decoder = args.layers_decoder
attention_heads = args.attention_heads
model_type = args.model_type
assert model_type in ["decoder", "encoder-decoder"]

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

retriever = DummyRetriever(default_k=k)

if model_type == "encoder-decoder":
    print("Creating encoder-decoder model...")
    model_encoder, model_decoder = createTransformerEncoderDecoder(args=args_model)
    ralm = ralmEncoderDecoder(model_encoder, model_decoder, retriever, index_scanner=index_scanner,
        batch_size=batch_size, retrieval_interval=retrieval_interval, retrieval_token_len=retrieval_token_len,
        k=k, device=device)
else:
    print("Creating decoder model...")
    model = createTransformerDecoder(args=args_model)
    ralm = ralmDecoder(model, retriever, index_scanner=index_scanner, batch_size=batch_size, retrieval_interval=retrieval_interval, device=device)

# functionality test
# ralm.single_step()
# ralm.multi_steps(num_step=seq_len - 1)
# ralm.reset_inference_state()

# batch processing
print("Start batch inference...")
for i in range(n_batch):
    print(f"Batch {i}")
    ralm.batch_inference(num_step=seq_len)

ralm.get_profiling()
ralm.print_profiling_stats()