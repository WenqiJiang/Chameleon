"""
Given multiple GPUs, each with a single model & single GPU & a single external retriever, 
    this process consumes the requests from the GPU processes, and forward them to the CPU search_server,
    the top-K results received from the CPU search_server are also multiplexed to the original GPU processes.
"""

import argparse 
import yaml

from ralm.coordinator.retriever_coordinator_server import RetrieveCoordinator

parser = argparse.ArgumentParser()

parser.add_argument('--coordinator_base_config', type=str, default="config/coordinator.yaml", help="address of the base configuration")

parser.add_argument('--search_server_host', type=str, default="127.0.0.1", help="space separated list of search_server hosts") # "127.0.0.1 127.0.0.1"
parser.add_argument('--search_server_port', type=str, default="9091", help="space separated list of search_server ports") # "9091 9092"
parser.add_argument('--ngpus', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_queries_per_gpu', type=int, default=10)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument("--nprobe", type=int, default=32)
parser.add_argument("--request_with_lists", type=int, default=0)

args = parser.parse_args()

# runtime params
search_server_host = args.search_server_host
search_server_port = args.search_server_port
ngpus = args.ngpus
batch_size = args.batch_size
num_queries_per_gpu = args.num_queries_per_gpu
k = args.k
dim = args.dim
nprobe = args.nprobe
request_with_lists = args.request_with_lists

# yaml inputs
local_host = None
local_port = None

# load configuration and put them as local variables
config_dict = {}
with open(args.coordinator_base_config, "r") as f:
    config_dict.update(yaml.safe_load(f))
locals().update(config_dict)

search_server_host = search_server_host.split(" ")
search_server_port = search_server_port.split(" ")
search_server_port = [int(port) for port in search_server_port]
assert len(search_server_host) == len(search_server_port)

coordinator = RetrieveCoordinator(
    search_server_host=search_server_host, search_server_port=search_server_port,
    local_host=local_host, local_port=local_port,
    ngpus=ngpus, batch_size=batch_size, num_queries_per_gpu=num_queries_per_gpu, 
    k=k, dim=dim, nprobe=nprobe, request_with_lists=request_with_lists)

# coordinator.start_dummy_answer() # return random results
coordinator.start() # return from the CPU search_server