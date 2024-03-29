"""
This script starts a single GPU process, and record the profiling statistics.

The profile saving only supports single-GPU experiments.

For multi-GPUs, please turn off `save_profiling`, we only need the throughput information, which can be found in the print messages.

The RALM performance is stored in a hierachical dictionary: 
    d_perf = d[model_name][architecture][retrieval_interval][batch_size]
        (we assume [dbname][index][k][nprobe] are tied to a certain model)
        d_perf['latency_ms'] = [latency_array_1 (batch 1), latency_array_2 (batch 2), ...],
            np array, shape = (batch_size, seq_len)
        d_perf['throughput_tokens_per_sec'] = [throughput_1, throughput_2, ...],
            np array, shape = (batch_size,)
"""

import os
import torch
import yaml
import time
import numpy as np

from ralm.ralm.ralm import ralmDecoder, ralmEncoderDecoder
from ralm.ralm.ralm_tiktok import ralmTikTokDecoder, ralmTikTokEncoderDecoder
from ralm.lm.get_model import createTransformerDecoder, createTransformerEncoderDecoder

from ralm.retriever.retriever import ExternalRetriever
from ralm.index_scanner.index_scanner import IndexScanner

from utils import save_obj, load_obj, generate_queries

import argparse 
parser = argparse.ArgumentParser()

# model configuration yaml
parser.add_argument('--model_base_config', type=str, default="config/Dec-S.yaml", help="address of the base configuration")
parser.add_argument('--coordinator_base_config', type=str, default="config/coordinator.yaml", help="address of the base configuration")

# runtime input
parser.add_argument('--retrieval_interval', type=int, default=1)
parser.add_argument('--request_with_lists', type=int, default=0, help="whether to scan index locally on the GPU")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_warmup_batch', type=int, default=1, help="number of warmup batches of inference to run")
parser.add_argument('--n_batch', type=int, default=1, help="number of batches of inference to run")
parser.add_argument('--use_gpu_id', type=int, default=None, help="the GPU ID to run the model")
parser.add_argument('--use_tiktok', type=int, default=0, help="0 or 1, whether to use tiktok")

parser.add_argument('--save_profiling', type=int, default=0, help="0 or 1, whether to save the profiling stats")
parser.add_argument('--architecture', type=str, default="1FPGA-8CPU", help="the search architecture")
parser.add_argument('--profile_dir', type=str, default='performance_results', help="directory to save the profiling stats")
parser.add_argument('--profile_fname', type=str, default='performance_results/ralm_profile', help="directory to save the profiling stats")

args = parser.parse_args()

# argparse inputs
retrieval_interval = args.retrieval_interval
request_with_lists = args.request_with_lists
batch_size = args.batch_size
n_warmup_batch = args.n_warmup_batch
n_batch = args.n_batch
use_gpu_id = args.use_gpu_id
use_tiktok = args.use_tiktok

save_profiling = args.save_profiling
architecture = args.architecture
profile_dir = args.profile_dir
profile_fname = args.profile_fname

print(save_profiling, profile_fname)

# yaml inputs
model_name = None
dim = None
layers_encoder = None
layers_decoder = None
attention_heads = None
model_type = None 

dbname = None
seq_len = None
k = None
retrieval_token_len = None
nlist = None
nprobe = None
use_coordinator = None
centroids_dir = None

local_host = None
local_port = None

config_dict = {}
with open(args.model_base_config, "r") as f:
    config_dict.update(yaml.safe_load(f))
with open(args.coordinator_base_config, "r") as f:
    config_dict.update(yaml.safe_load(f))
locals().update(config_dict)

assert model_type in ["decoder", "encoder-decoder"]
assert use_tiktok in [0, 1]

if save_profiling:
    if os.path.exists(os.path.join(profile_dir, profile_fname + '.pkl')):
        print("Loading existing profiling stats...", flush=True)
        perf_d = load_obj(profile_dir, profile_fname)
    else:
        print("Creating a new profiling stats...", flush=True)
        perf_d = {}

# load query
if dbname in ['RALM-S1000M', 'RALM-S2000M', 'RALM-L1000M', 'RALM-L2000M']:
	print("Loading queries...", flush=True)
	query_set = generate_queries(dbname, 10000, 1)[0]
else:
	print(f"Unknown dbname {dbname}, no pre-generated queries", flush=True)
	query_set = None

# load centroids from pure binary float32 file
if centroids_dir is not None:
	centroids = np.fromfile(centroids_dir, dtype=np.float32).reshape(nlist, dim)
	print("centroids_dir: {}".format(centroids_dir))	
	print(centroids.shape)
else:
	centroids = None

# instantiate the retriever
retriever = ExternalRetriever(default_k=k, host=local_host, port=local_port, batch_size=batch_size, dim=dim)

# instantiate the index scanner
# TODO: also support CPU index scanner    
if request_with_lists:
    print("Creating GPU index scanner...", flush=True)
    index_scanner = IndexScanner(dim=dim, nlist=nlist, nprobe=nprobe, centroids=centroids, use_gpu_id=use_gpu_id)
else:
    index_scanner = None
    

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
print(f"Executing on {device}", flush=True)

if use_tiktok:
    print("Using tiktok scheduling...", flush=True)
else:
    print("Disabled tiktok scheduling...", flush=True)

if model_type == "encoder-decoder":
    print("Creating encoder-decoder model...", flush=True)
    model_encoder, model_decoder = createTransformerEncoderDecoder(args=args_model)
    if use_tiktok:
        ralm = ralmTikTokEncoderDecoder(model_encoder, model_decoder, retriever, index_scanner=index_scanner, query_set=query_set,
            batch_size=batch_size, retrieval_interval=retrieval_interval, retrieval_token_len=retrieval_token_len, 
            k=k, device=device, use_coordinator=use_coordinator)
    else:
        ralm = ralmEncoderDecoder(model_encoder, model_decoder, retriever, index_scanner=index_scanner, query_set=query_set,
            batch_size=batch_size, retrieval_interval=retrieval_interval, retrieval_token_len=retrieval_token_len, 
            k=k, device=device, use_coordinator=use_coordinator)
else:
    print("Creating decoder model...", flush=True)
    model_decoder = createTransformerDecoder(args=args_model)
    if use_tiktok:
        ralm = ralmTikTokDecoder(model_decoder, retriever, index_scanner=index_scanner, query_set=query_set, batch_size=batch_size, 
            retrieval_interval=retrieval_interval, device=device, use_coordinator=use_coordinator)
    else:
        ralm = ralmDecoder(model_decoder, retriever, index_scanner=index_scanner, query_set=query_set, batch_size=batch_size, 
            retrieval_interval=retrieval_interval, device=device, use_coordinator=use_coordinator)

# warmup
print("Start warmup...")
for i in range(n_warmup_batch):
    print(f"Batch {i}", flush=True)
    ralm.batch_inference(num_step=seq_len)

# batch processing
print("Start batch inference...", flush=True)
# all time in ms
list_time_ms_model = []
list_time_ms_model_encoder = []
list_time_ms_model_decoder = []
list_time_ms_retriever = []
list_time_ms_step = []
list_throughput_tokens_per_sec = []

for i in range(n_batch):
    print(f"Batch {i}", flush=True)
    t_start = time.time()
    ralm.batch_inference(num_step=seq_len)
    t_end = time.time()
    if model_type == "encoder-decoder":
        time_model, time_model_encoder, time_model_decoder, time_retriever, time_step = ralm.get_profiling()
        list_time_ms_model.append(np.array(time_model) * 1000)
        list_time_ms_model_encoder.append(np.array(time_model_encoder) * 1000)
        list_time_ms_model_decoder.append(np.array(time_model_decoder) * 1000)
        list_time_ms_retriever.append(np.array(time_retriever) * 1000)
        list_time_ms_step.append(np.array(time_step) * 1000)
    else:
        time_model, time_retriever, time_step = ralm.get_profiling()
        list_time_ms_model.append(np.array(time_model) * 1000)
        list_time_ms_retriever.append(np.array(time_retriever) * 1000)
        list_time_ms_step.append(np.array(time_step) * 1000)
    ralm.print_profiling_stats()
    tokens_per_sec = batch_size * seq_len / (t_end - t_start)
    list_throughput_tokens_per_sec.append(tokens_per_sec)
print("Throughput (tokens per sec):", list_throughput_tokens_per_sec, flush=True)

list_time_ms_model = np.array(list_time_ms_model)
list_time_ms_model_encoder = np.array(list_time_ms_model_encoder)
list_time_ms_model_decoder = np.array(list_time_ms_model_decoder)
list_time_ms_retriever = np.array(list_time_ms_retriever)
list_time_ms_step = np.array(list_time_ms_step)
list_throughput_tokens_per_sec = np.array(list_throughput_tokens_per_sec)

assert list_time_ms_step.shape == (n_batch, seq_len)
assert list_throughput_tokens_per_sec.shape == (n_batch,)

if save_profiling:
    # profile_dict = {}
    # # each entry is a list with length n_batch
    # if model_type == "encoder-decoder":
    #     profile_dict["time_model"] = list_time_ms_model
    #     profile_dict["time_model_encoder"] = list_time_ms_model_encoder
    #     profile_dict["time_model_decoder"] = list_time_ms_model_decoder
    #     profile_dict["time_retriever"] = list_time_ms_retriever
    #     profile_dict["time_step"] = list_time_ms_step
    # else:
    #     profile_dict["time_model"] = list_time_ms_model
    #     profile_dict["time_retriever"] = list_time_ms_retriever
    #     profile_dict["time_step"] = list_time_ms_step

    if model_name not in perf_d:
        perf_d[model_name] = {}
    if architecture not in perf_d[model_name]:
        perf_d[model_name][architecture] = {}
    if retrieval_interval not in perf_d[model_name][architecture]:
        perf_d[model_name][architecture][retrieval_interval] = {}
    if batch_size not in perf_d[model_name][architecture][retrieval_interval]:
        perf_d[model_name][architecture][retrieval_interval][batch_size] = {}


    perf_d[model_name][architecture][retrieval_interval][batch_size]["latency_ms"] = list_time_ms_step
    perf_d[model_name][architecture][retrieval_interval][batch_size]["throughput_tokens_per_sec"] = list_throughput_tokens_per_sec
    assert len(list_time_ms_step) == len(list_throughput_tokens_per_sec)
    for i in range(n_batch):
        assert len(list_time_ms_step[i]) == seq_len
    
    save_obj(perf_d, profile_dir, profile_fname)
    print("Profiling stats saved to", profile_dir, profile_fname, flush=True)
