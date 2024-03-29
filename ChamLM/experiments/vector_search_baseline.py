"""
Evaluate latency and throughput of various datasets, indexes, and architectures.

Example Usage:
    python vector_search_baseline.py --load_dict 1 --run_latency_experiments 1 --run_throughput_experiments 1 --latency_batch_num 100 --throughput_batch_size 10000 --throughput_runs 5 --start_gpu_id 0

Latency result format (in dict): [dbname][index][architecture][k][nprobe][batch_size] = latency_array (ms)
    dbname example: 'SIFT1000M'
    index example: 'IVF32768,PQ16'
    nprobe example: 32
    batch_size example: 64
    architecture example: '16CPU-1GPU' or '32CPU', here the number means the number of CPU cores and GPU cards
    latency_array example: np.array([1.5, 3.4, ...]) (ms)

    
Throughput result format (in dict): [dbname][index][architecture][k][nprobe] = throughput_array (QPS)
"""

import argparse 
import os
import time
import gc
import numpy as np

from ralm.retriever.faiss_retriever import LocalFaissRetriever
from utils import save_obj, load_obj, parse_arch, generate_queries

parser = argparse.ArgumentParser()

parser.add_argument("--load_dict", type=int, default=1, help="load saved dict or not as the intial dict")
parser.add_argument("--run_latency_experiments", type=int, default=1, help="run latency experiments or not")
parser.add_argument("--run_throughput_experiments", type=int, default=1, help="run throughput experiments or not")
parser.add_argument("--latency_batch_num", type=int, default=100, help="number of batches to run for latency experiments")
parser.add_argument("--throughput_batch_size", type=int, default=10000, help="batch size for throughput experiments")
parser.add_argument("--throughput_runs", type=int, default=5, help="number of runs for throughput experiments")
parser.add_argument("--start_gpu_id", type=int, default=0, help="the starting GPU ID to run the model")

args = parser.parse_args()
load_dict = args.load_dict
run_latency_experiments = args.run_latency_experiments
run_throughput_experiments = args.run_throughput_experiments
latency_batch_num = args.latency_batch_num
throughput_batch_size = args.throughput_batch_size
throughput_runs = args.throughput_runs
start_gpu_id = args.start_gpu_id

base_dir = '/mnt/scratch/wenqi/Faiss_experiments/trained_CPU_indexes/'
dbname_index_dim_arch_tuples = [
	# # Deep1000M, PQ16, 16 GB, compared with 1 FPG
    # ('Deep1000M', 'IVF32768,PQ16', 96, '8CPU'),   
    # ('Deep1000M', 'IVF32768,PQ16', 96, '8CPU-1GPU'),   
    # ('Deep1000M', 'IVF32768,PQ16', 96, '16CPU'), 

	# # SIFT1000M, PQ16, 16 GB, compared with 1 FPG
    # ('SIFT1000M', 'IVF32768,PQ16', 128, '8CPU'),   
    # ('SIFT1000M', 'IVF32768,PQ16', 128, '8CPU-1GPU'),   
    # ('SIFT1000M', 'IVF32768,PQ16', 128, '16CPU'),   
       
	# RALM-S1000M, PQ32, 32 GB, compared with 1 FPG
    ('RALM-S1000M', 'IVF32768,PQ32', 512, '8CPU'),   
    ('RALM-S1000M', 'IVF32768,PQ32', 512, '8CPU-1GPU'),   
    ('RALM-S1000M', 'IVF32768,PQ32', 512, '16CPU'),   
       
	# # RALM-L1000M, PQ64, 64 GB, compared with 2 FPG
    # ('RALM-L1000M', 'IVF32768,PQ64', 1024, '16CPU'),   
    # ('RALM-L1000M', 'IVF32768,PQ64', 1024, '16CPU-1GPU'),   
    # ('RALM-L1000M', 'IVF32768,PQ64', 1024, '32CPU'),   

	# # RALM-S2000M, PQ32, 72 GB, compared with 2 FPG
    # ('RALM-S2000M', 'IVF32768,PQ32', 512, '16CPU'),   
    # ('RALM-S2000M', 'IVF32768,PQ32', 512, '16CPU-1GPU'),   
    # ('RALM-S2000M', 'IVF32768,PQ32', 512, '32CPU'),   
       
	# # RALM-L2000M, PQ64, 128 GB, compared with 4 FPG
    # ('RALM-L2000M', 'IVF32768,PQ64', 1024, '16CPU'),   
    # ('RALM-L2000M', 'IVF32768,PQ64', 1024, '16CPU-1GPU'),   
    # ('RALM-L2000M', 'IVF32768,PQ64', 1024, '32CPU'),   

]

list_k = [100]
list_nprobe = [32]
list_batch_size = [1, 2, 4, 8, 16, 32, 64, 128]

dict_dir  = './performance_results'
latency_dict_name = 'vector_search_latency_CPU_GPU_dict'
throughput_dict_name = 'vector_search_throughput_CPU_GPU_dict'

if run_latency_experiments:
    if load_dict and os.path.exists(os.path.join(dict_dir,latency_dict_name + '.pkl')):
        print("Loading latency dict...")
        latency_dict = load_obj(dict_dir,latency_dict_name)
    else:
        print("Create a new latency dict...""")
        latency_dict = {}
if run_throughput_experiments:
    if load_dict and os.path.exists(os.path.join(dict_dir,throughput_dict_name + '.pkl')):
        print("Loading throughput dict...")
        throughput_dict = load_obj(dict_dir,throughput_dict_name)
    else:
        print("Create a new throughput dict...""")
        throughput_dict = {}

for dbname, index_key, dim, arch in dbname_index_dim_arch_tuples:
    
    if dbname not in latency_dict:
        latency_dict[dbname] = {}
    if index_key not in latency_dict[dbname]:
        latency_dict[dbname][index_key] = {}
    if arch not in latency_dict[dbname][index_key]:
        latency_dict[dbname][index_key][arch] = {}
    if dbname not in throughput_dict:
        throughput_dict[dbname] = {}
    if index_key not in throughput_dict[dbname]:
        throughput_dict[dbname][index_key] = {}
    if arch not in throughput_dict[dbname][index_key]:
        throughput_dict[dbname][index_key][arch] = {}
    
    index_dir = os.path.join(base_dir, f'bench_cpu_{dbname}_{index_key}')
    
    print(f'Start {dbname} {index_key} {arch}')
    # set up retriever
    device, omp_threads, ngpu = parse_arch(arch)
    retriever = LocalFaissRetriever(index_dir=index_dir, dbname=dbname, index_key=index_key, device=device, ngpu=ngpu, start_gpu_id=start_gpu_id, omp_threads=omp_threads)

    # generate latency_query
    max_query_num = latency_batch_num * list_batch_size[-1]

    for k in list_k:
        if k not in latency_dict[dbname][index_key][arch]:
            latency_dict[dbname][index_key][arch][k] = {}
        if k not in throughput_dict[dbname][index_key][arch]:
            throughput_dict[dbname][index_key][arch][k] = {}
        
        retriever.default_k = k
        print(f'k = {k}')

        for nprobe in list_nprobe:
            if nprobe not in latency_dict[dbname][index_key][arch][k]:
                latency_dict[dbname][index_key][arch][k][nprobe] = {}
            if nprobe not in throughput_dict[dbname][index_key][arch][k]:
                throughput_dict[dbname][index_key][arch][k][nprobe] = {}
            print(f'nprobe = {nprobe}')
            
            retriever.set_nprobe(nprobe)

            if run_latency_experiments:
                # Latency experiments of different batch sizes
                for batch_size in list_batch_size:
                    if batch_size not in latency_dict[dbname][index_key][arch][k][nprobe]:
                        latency_dict[dbname][index_key][arch][k][nprobe][batch_size] = {}
                    print(f'batch_size = {batch_size}')

                    warmup_query = np.random.rand(batch_size, dim).astype('float32')
                    retriever.retrieve(query=warmup_query, nprobe=nprobe, k=k)
                    
                    latency_query_list = generate_queries(dbname, batch_size, latency_batch_num)

                    time_list = []
                    for i in range(latency_batch_num):
                        t_start = time.time()
                        retriever.retrieve(query=latency_query_list[i], nprobe=nprobe, k=k)
                        t_end = time.time()
                        time_list.append(t_end - t_start)
                    
                    time_list_ms = np.array(time_list) * 1000
                    latency_dict[dbname][index_key][arch][k][nprobe][batch_size] = time_list_ms

                    print("Average latency: {:.2f} ms".format(np.average(time_list_ms)))

            throughput_query = generate_queries(dbname, throughput_batch_size, 1)
            if run_throughput_experiments:
                time_list = []
                for n_run in range(throughput_runs):
                    t_start = time.time()
                    retriever.retrieve(query=throughput_query[0], nprobe=nprobe, k=k)
                    t_end = time.time()
                    time_list.append(t_end - t_start)
                qps_list = throughput_batch_size / np.array(time_list) 
                throughput_dict[dbname][index_key][arch][k][nprobe] = qps_list
                print("Average QPS: {:.2f}".format(np.average(qps_list)))

    # deallocate memory of the retriever
    del retriever
    gc.collect()

    print('Saving dict of the current db, index, arc...')
    save_obj(latency_dict, dict_dir, latency_dict_name)
    save_obj(throughput_dict, dict_dir, throughput_dict_name)
        
print("Done!")