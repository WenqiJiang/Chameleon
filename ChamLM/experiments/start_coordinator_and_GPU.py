"""
The process loads a configuration file of the model, takes several runtime parameters, 
    and starts the coordinator and the GPU(s).

Example usage:
    python start_coordinator_and_GPU.py --model_base_config config/Dec-S.yaml --coordinator_base_config config/coordinator.yaml \
        --search_server_host "127.0.0.1" --search_server_port "9091" \
        --retrieval_interval 1 --request_with_lists 0 --batch_size 32 --n_warmup_batch 1 --n_batch 1 \
        --ngpus 1 --start_gpu_id 0 --use_tiktok 0 --save_profiling 1

"""

import yaml
import os
import time

import argparse 
parser = argparse.ArgumentParser()

# model configuration yaml
parser.add_argument('--model_base_config', type=str, default="config/Dec-S.yaml", help="address of the base configuration")
parser.add_argument('--coordinator_base_config', type=str, default="config/coordinator.yaml", help="address of the base configuration")

# runtime input (server)
parser.add_argument('--search_server_host', type=str, default="127.0.0.1", help="space separated list of search_server hosts") # "127.0.0.1 127.0.0.1"
parser.add_argument('--search_server_port', type=str, default="9091", help="space separated list of search_server ports") # "9091 9092"
# runtime input (GPU)
parser.add_argument('--retrieval_interval', type=int, default=1)
parser.add_argument('--request_with_lists', type=int, default=0, help="whether to scan index locally on the GPU")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_warmup_batch', type=int, default=1, help="number of warmup batches of inference to run")
parser.add_argument('--n_batch', type=int, default=1, help="number of batches of inference to run")
parser.add_argument('--ngpus', type=int, default=1)
parser.add_argument('--start_gpu_id', type=int, default=0, help="the first GPU ID to run the model")
parser.add_argument('--use_tiktok', type=int, default=0, help="0 or 1, whether to use tiktok")
parser.add_argument('--save_profiling', type=int, default=0, help="0 or 1, whether to save the profiling stats")
parser.add_argument('--architecture', type=str, default="1FPGA-8CPU", help="the search architecture")
parser.add_argument('--profile_dir', type=str, default='performance_results', help="directory to save the profiling stats")
parser.add_argument('--profile_fname', type=str, default='RALM_latency_throughput', help="directory to save the profiling stats")

args = parser.parse_args()

# runtime input
search_server_host = args.search_server_host
search_server_port = args.search_server_port

retrieval_interval = args.retrieval_interval
request_with_lists = args.request_with_lists
batch_size = args.batch_size
n_warmup_batch = args.n_warmup_batch
n_batch = args.n_batch
ngpus = args.ngpus
start_gpu_id = args.start_gpu_id
use_tiktok = args.use_tiktok
save_profiling = args.save_profiling
architecture = args.architecture
profile_dir = args.profile_dir
profile_fname = args.profile_fname

# yaml inputs
model_name = None
dim = None
layers_encoder = None
layers_decoder = None
attention_heads = None
model_type = None 

seq_len = None
k = None
retrieval_token_len = None
nlist = None
nprobe = None
use_coordinator = None

model_base_config = args.model_base_config
coordinator_base_config = args.coordinator_base_config

config_dict = {}
with open(args.model_base_config, "r") as f:
    config_dict.update(yaml.safe_load(f))
with open(args.coordinator_base_config, "r") as f:
    config_dict.update(yaml.safe_load(f))
locals().update(config_dict)

print("Please make sure the server is already running...")

# Start the coordinator
n_retrieve_per_seq = int(seq_len / retrieval_interval)
if use_tiktok:
    n_retrieve_per_seq *= 2
num_queries_per_gpu = n_retrieve_per_seq * (n_batch + n_warmup_batch)

log_coord = 'logs/coordinator.log'
cmd_coord = f"python coordinator_process.py --coordinator_base_config {coordinator_base_config} " \
    f"--search_server_host {search_server_host} --search_server_port {search_server_port} --ngpus {ngpus} " \
    f"--batch_size {batch_size} --num_queries_per_gpu {num_queries_per_gpu} --k {k} --dim {dim} " \
    f"--nprobe {nprobe} --request_with_lists {request_with_lists} > {log_coord} 2>&1 &"

print("\nRunning the coordinator...")
print(cmd_coord)
os.system(cmd_coord)
time.sleep(5) # wait for the coordinator to start

# Start the GPU processes
print("\nRunning the GPU processes...")
for i in range(ngpus):
    log_gpu = f'logs/gpu_{i}.log'
    # profile_fname = f'{model_name}_retrieval_interval_{retrieval_interval}_' \
    #     f'request_with_lists_{request_with_lists}_' \
    #     f'batch_size_{batch_size}_n_batch_{n_batch}_use_tiktok_{use_tiktok}_' \
    #     f'ngpus_{ngpus}_gpu_{i}'
    cmd_gpu = f"python GPU_process.py --model_base_config {model_base_config} " \
        f"--coordinator_base_config {coordinator_base_config} --retrieval_interval {retrieval_interval} " \
        f"--request_with_lists {request_with_lists} --batch_size {batch_size} --n_warmup_batch {n_warmup_batch} " \
        f"--n_batch {n_batch} --use_gpu_id {start_gpu_id + i} --use_tiktok {use_tiktok} " \
        f"--save_profiling {save_profiling} --architecture {architecture} --profile_dir {profile_dir} --profile_fname {profile_fname} " \
        f" > {log_gpu} 2>&1 &"
    print(cmd_gpu)
    os.system(cmd_gpu)

print("")
print("\nLauched the processes, to kill them, run the following commands:")
print("ps aux | grep 'python coordinator_process.py' | awk '{print $2}' | xargs -i kill -9 {}")
print("ps aux | grep 'python GPU_process.py' | awk '{print $2}' | xargs -i kill -9 {}")