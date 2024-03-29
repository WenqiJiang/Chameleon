"""
Evaluate latency or throughput of FPGA vector search on various datasets, by starting the Faiss index server.

Before starting this script, make sure that the search server (FPGA coordinator) is already running.

Example Usage (latency):
    python vector_search_FPGA.py --search_base_config config/search_SIFT1000M.yaml \
        --search_server_host "127.0.0.1" --search_server_port 9091 \
        --batch_size 32 --n_warmup_batch 5 --n_batch 100 --architecture "1FPGA-8CPU" --use_gpu_id 0 --mode "latency" --load_dict 1
    
Example Usage (throughput):
    python vector_search_FPGA.py --search_base_config config/search_SIFT1000M.yaml \
        --search_server_host "127.0.0.1" --search_server_port 9091 \
        --batch_size 128 --n_warmup_batch 5 --n_batch 20 --architecture "1FPGA-8CPU" --use_gpu_id 0 --mode "throughput" --throughput_runs 5 --load_dict 1
        
Latency result format (in dict): [dbname][index][architecture][k][nprobe][batch_size][latency_array (ms)]
    dbname example: 'SIFT1000M'
    index example: 'IVF32768,PQ16'
    nprobe example: 32
    batch_size example: 64
    architecture example: '1FPGA-1GPU' or '2FPGA-8CPU', here the number means the number of CPU cores and FPGA cards
    latency_array example: np.array([1.5, 3.4, ...]) (ms)
    
Throughput result format (in dict): [dbname][index][architecture][k][nprobe][batch_size][throughput_array (QPS)]
"""

import argparse 
import os
import time
import yaml
import numpy as np

from ralm.index_scanner.index_server import IndexServer
from utils import save_obj, load_obj, generate_queries

parser = argparse.ArgumentParser()
parser.add_argument('--search_base_config', type=str, default="config/search_SIFT1000M.yaml", help="directory of the base configuration")

parser.add_argument('--search_server_host', type=str, default="127.0.0.1", help="to the FPGA coordinator")
parser.add_argument('--search_server_port', type=int, default=9091, help="to the FPGA coordinator") 
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_warmup_batch', type=int, default=1, help="number of warmup batches of search to run")
parser.add_argument('--n_batch', type=int, default=100, help="number of batches of search to run, in both latency and throughput experiments")
parser.add_argument('--architecture', type=str, default="1FPGA-8CPU") 
parser.add_argument('--use_gpu_id', type=int, default=None, help="the first GPU ID to run the model")
parser.add_argument('--mode', type=str, default='latency', help="latency or throughput, determines whether to use tiktok")
parser.add_argument("--throughput_runs", type=int, default=5, help="number of runs for throughput experiments, each will run tik-tok in n_batch batches")
parser.add_argument("--load_dict", type=int, default=1, help="load saved dict or not as the intial dict")

args = parser.parse_args()


# runtime input
search_server_host = args.search_server_host
search_server_port = args.search_server_port
batch_size = args.batch_size
n_warmup_batch = args.n_warmup_batch
n_batch = args.n_batch
use_gpu_id = args.use_gpu_id
architecture = args.architecture
mode = args.mode
load_dict = args.load_dict
assert mode in ['latency', 'throughput']
throughput_runs = args.throughput_runs

if 'GPU' in architecture:
    device = 'gpu'
    omp_threads = None
    print("Index device: GPU")
else:
    device = 'cpu'
    omp_threads = int(architecture.split('-')[-1].split('CPU')[0])
    print("Index device: CPU, omp_threads: {}".format(omp_threads))

# yaml inputs
dbname = None
index_key = None
dim = None
k = None
nlist = None
nprobe = None
centroids_dir = None
config_dict = {}
with open(args.search_base_config, "r") as f:
    config_dict.update(yaml.safe_load(f))
locals().update(config_dict)

# load dict
dict_dir  = './performance_results'
latency_dict_name = 'vector_search_latency_FPGA_dict'
throughput_dict_name = 'vector_search_throughput_FPGA_dict'
if mode == 'latency':
    if load_dict and os.path.exists(os.path.join(dict_dir,latency_dict_name + '.pkl')):
        print("Loading latency dict...")
        latency_dict = load_obj(dict_dir,latency_dict_name)
    else:
        print("Create a new latency dict...""")
        latency_dict = {}
elif mode == 'throughput':
    if load_dict and os.path.exists(os.path.join(dict_dir,throughput_dict_name + '.pkl')):
        print("Loading throughput dict...")
        throughput_dict = load_obj(dict_dir,throughput_dict_name)
    else:
        print("Create a new throughput dict...""")
        throughput_dict = {}

# assert architecture
if dbname == 'SIFT1000M' or dbname == 'Deep1000M' or dbname == 'RALM-S1000M':
    assert architecture in ["1FPGA-1GPU", "1FPGA-8CPU"]
elif dbname == 'RALM-L1000M' or dbname == 'RALM-S2000M':
    assert architecture in ["2FPGA-1GPU", "2FPGA-8CPU"]
elif dbname == 'RALM-L2000M':
    assert architecture in ["4FPGA-1GPU", "4FPGA-8CPU"]
else:
    raise ValueError("Unknown dataset")

# load centroids from pure binary float32 file
if centroids_dir is not None:
	centroids = np.fromfile(centroids_dir, dtype=np.float32).reshape(nlist, dim)
	print("centroids_dir: {}".format(centroids_dir))	
	print(centroids.shape)
else:
	centroids = None

index_server = IndexServer(
    host=search_server_host, port=search_server_port, dim=dim, nlist=nlist, nprobe=nprobe, default_k=k, batch_size=batch_size,
    centroids=centroids, device=device, use_gpu_id=use_gpu_id, omp_threads=omp_threads)


warm_up_query_list = generate_queries(dbname, batch_size, n_warmup_batch)
query_list = generate_queries(dbname, batch_size, n_batch)

print("Starting the experiments...")
if mode == 'latency':

    if dbname not in latency_dict:
        latency_dict[dbname] = {}
    if index_key not in latency_dict[dbname]:
        latency_dict[dbname][index_key] = {}
    if architecture not in latency_dict[dbname][index_key]:
        latency_dict[dbname][index_key][architecture] = {}
    if k not in latency_dict[dbname][index_key][architecture]:
        latency_dict[dbname][index_key][architecture][k] = {}
    if nprobe not in latency_dict[dbname][index_key][architecture][k]:
        latency_dict[dbname][index_key][architecture][k][nprobe] = {}
    if batch_size not in latency_dict[dbname][index_key][architecture][k][nprobe]:
        latency_dict[dbname][index_key][architecture][k][nprobe][batch_size] = {}

    # warm up
    index_server.search_multi_batch(warm_up_query_list, nprobe, k)
    print("Finish warm up")

    # run and profile
    indices, distances = index_server.search_multi_batch(query_list, nprobe, k)
    latency_list_sec, throughput_batches_per_sec = index_server.get_profiling()
    latency_list_ms = np.array(latency_list_sec) * 1000
    latency_dict[dbname][index_key][architecture][k][nprobe][batch_size] = latency_list_ms

    print("Average latency: {:.2f} ms".format(np.average(latency_list_ms)))

    print('Saving dict of the current run...')
    save_obj(latency_dict, dict_dir, latency_dict_name)

elif mode == 'throughput':
    
    if dbname not in throughput_dict:
        throughput_dict[dbname] = {}
    if index_key not in throughput_dict[dbname]:
        throughput_dict[dbname][index_key] = {}
    if architecture not in throughput_dict[dbname][index_key]:
        throughput_dict[dbname][index_key][architecture] = {}
    if k not in throughput_dict[dbname][index_key][architecture]:
        throughput_dict[dbname][index_key][architecture][k] = {}
    if nprobe not in throughput_dict[dbname][index_key][architecture][k]: 
        throughput_dict[dbname][index_key][architecture][k][nprobe] = {}

    # warm up
    index_server.search_multi_batch_tiktok(warm_up_query_list, nprobe, k)
    print("Finish warm up")

    # run and profile
    qps_list = []
    for i in range(throughput_runs):
        indices, distances = index_server.search_multi_batch_tiktok(query_list, nprobe, k)
        latency_list_sec, throughput_batches_per_sec = index_server.get_profiling()
        qps_list.append(throughput_batches_per_sec * batch_size)
    throughput_dict[dbname][index_key][architecture][k][nprobe] = qps_list
    print("Average QPS: {:.2f}".format(np.average(qps_list)))

    print('Saving dict of the current run...')
    save_obj(throughput_dict, dict_dir, throughput_dict_name)