"""
Example Usage:

    cd ralm/retriever

    # queries without list IDs
    python faiss_server.py --dbname RALM-S1M --index_key IVF32768,PQ32 --index_dir /mnt/scratch/wenqi/Faiss_experiments/trained_CPU_indexes/bench_cpu_RALM-S1M_IVF32768,PQ32/ \
        --host 127.0.0.1 --port 9090 --batch_size 32 --dim 512 --k 10 --nprobe 32 --request_with_lists 0 --device cpu-gpu --ngpu 1
    
    # queries with list IDs
    python faiss_server.py --dbname RALM-S1M --index_key IVF32768,PQ32 --index_dir /mnt/scratch/wenqi/Faiss_experiments/trained_CPU_indexes/bench_cpu_RALM-S1M_IVF32768,PQ32/ \
        --host 127.0.0.1 --port 9090 --batch_size 32 --dim 512 --k 10 --nprobe 32 --request_with_lists 1 --device cpu --ngpu 1    
"""

import torch
import os 
import faiss
import numpy as np
import socket

from ralm.server.server import BaseServer
from ralm.retriever.serialization_utils import request_message_length, request_message_length_with_lists, decode_request, decode_request_with_lists, encode_answer
from typing import Optional
from faiss.contrib.ivf_tools import search_preassigned

class FaissServer(BaseServer):

    def __init__(self, 
            # Server settings
            host: Optional[str] = None, port: Optional[int] = None, 
            # Faiss settings
            index_dir : Optional[str] = "/mnt/scratch/wenqi/Faiss_experiments/trained_CPU_indexes/bench_cpu_SIFT1000M_IVF16384,PQ16", 
            dbname : Optional[str] = "SIFT1000M", 
            index_key : Optional[str] = "IVF16384,PQ16", 
            default_k : Optional[int] = 10, nprobe : Optional[int] = 1,
            batch_size : Optional[int] = 1, dim : Optional[int] = 512,
            request_with_lists : Optional[int] = 0,
            device : Optional[str] = 'cpu', # 'cpu' or 'gpu' or 'cpu-gpu'
            # ngpu, use_float16, use_precomputed_tables are used only when device == 'gpu 
            ngpu : Optional[int] = 1, 
            omp_threads : Optional[int] = None,
            use_float16 : Optional[bool] = True, 
            use_precomputed_tables : Optional[bool] = True
            ):
        
        # Intialize Faiss
        self.index_dir = index_dir
        self.dbname = dbname
        self.index_key = index_key 
        self.default_k = default_k
        self.nprobe = nprobe
        self.batch_size = batch_size
        self.dim = dim
        self.request_with_lists = request_with_lists
        self.device = device
        if request_with_lists == 1:
            assert device == 'cpu' # GPU-only or CPU-GPU only for request without lists
        
        self.index = self.get_populated_index()
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)
        self.set_nprobe(nprobe)
        
        assert self.dim == self.index.d

        if device == 'gpu':
            self.entire_index_to_gpu(ngpu=ngpu)
        elif device == 'cpu-gpu':
            self.IVF_index_to_gpu(ngpu=ngpu)

        if self.retrieve_with_lists == 1 or self.device == 'cpu-gpu':
            """
            search_preassigned must be set with a proper parallel mode:
                if call search itself, the parallel for will be outside the search_preassigned function
                    https://github.com/facebookresearch/faiss/blob/v1.7.2/faiss/IndexIVF.cpp#L350
                if parallel_mode = 0 (default), the do_parallel will be set as false (single thread):
                    https://github.com/facebookresearch/faiss/blob/v1.7.2/faiss/IndexIVF.cpp#L416

            mode 3 is the fastest mode in the test
            """
            self.index.parallel_mode = 3

        if omp_threads is not None and device != 'gpu':
            print("WARNING: setting omp thread number to", omp_threads, 
                  ", please make sure only one Faiss object exists in the current process, "
                  "otherwise it can affect the performance of other Faiss objects.")
            self.omp_threads = omp_threads
            faiss.omp_set_num_threads(self.omp_threads)  

        # Start socket 
        self.HOST = socket.gethostbyname(
            host) if host else socket.gethostbyname(socket.gethostname())
        self.PORT = port if port else 9090
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server.bind((self.HOST, self.PORT))
        if self.request_with_lists:
            self.query_msg_len = request_message_length_with_lists(self.batch_size, self.dim, self.nprobe)
        else:
            self.query_msg_len = request_message_length(self.batch_size, self.dim)
            
        print("Server start listening...", flush=True)
        print("Query message length is {} bytes".format(self.query_msg_len), flush=True)
        self.server.listen(5)  # limit to 5 parallel connections

    def get_populated_index(self):
        """
        Load trained faiss index.
        """
        filename = os.path.join(self.index_dir, 
              f"{self.dbname}_{self.index_key}_populated.index")
        print("loading", filename, flush=True)
        index = faiss.read_index(filename)

        return index
    
    def entire_index_to_gpu(self, ngpu : Optional[int] = 1, 
        use_float16 : Optional[bool] = True, 
        use_precomputed_tables : Optional[bool] = True):
        """
        Move the index to (multiple) GPUs
            ngpu: multiple GPUs
            tempmem: the temporary memory per GPU in bytes, e.g., 1.5 GB = 1536*1024*1024
        """
        
        # doc: https://github.com/facebookresearch/faiss/blob/main/faiss/python/gpu_wrappers.py

        # # doc: https://faiss.ai/cpp_api/struct/structfaiss_1_1gpu_1_1ToGpuCloner.html
        # co = faiss.GpuMultipleClonerOptions()
        # co.useFloat16 = use_float16
        # co.useFloat16CoarseQuantizer = False
        # co.usePrecomputed = use_precomputed_tables
        # co.indicesOptions = 0
        # co.verbose = True
        # co.shard = True    # the replicas will be made "manually"
        # assert co.shard_type in (0, 1, 2)
        
        print("Moving the entire index to the GPU...", flush=True)
        self.index = faiss.index_cpu_to_gpus_list(index=self.index, ngpu=ngpu)

        """
        The older faiss interface does not work anymore... segment fault during cpu to gpu
        vres, vdev = self.make_vres_vdev(ngpu=ngpu, start_gpu=start_gpu, tempmem=tempmem)
        # print(vres, vdev)
        # self.index = faiss.index_cpu_to_gpu_multiple(
        # 	vres, vdev, self.index, co)
        """
        print("Index to gpu finished.", flush=True)

    def IVF_index_to_gpu(self, ngpu : Optional[int] = 1):
        """
        Only copy the IVF index (in IVF-PQ) to the GPU
        """
        print("Moving only the IVF index to the GPU...", flush=True)
        self.IVF_index = faiss.downcast_index(self.index.quantizer)
        self.IVF_index = faiss.index_cpu_to_gpus_list(index=self.IVF_index, ngpu=ngpu)
        print("Index to gpu finished.", flush=True)


    def set_nprobe(self, nprobe : int):

        self.nprobe = nprobe
        self.index.nprobe = nprobe
        # param = "nprobe={}".format(nprobe*1.0)
        # self.ps.set_index_parameters(self.index, 'nprobe', nprobe * 1.0)
        # self.ps.set_index_parameters(self.index, param)

    def retrieve(self, query : np.array, nprobe : Optional[int] = None, k : Optional[int] = None):
        """
        Input: 
            query: np.array, shape: (batch_size, hidden_dim)
            nprobe: number of clusters to scan
            k: number of results to return  

        Return some dummy output as a dict
            {"id" : [1, 10, 56, ...],  "dist": [.2, 1.2, 9.7, ...]}
            "id" : int64 -> np array with shape batch_size * k
            "dist" : float32 -> np array with shape batch_size * k
        """
        if k is None: 
            k = self.default_k
        if nprobe is None:
            nprobe = self.nprobe
        else:
            self.set_nprobe(nprobe)

        nq, dim = query.shape
        assert dim == self.dim

        # I = np.empty((nq, k), dtype='int64')
        # D = np.empty((nq, k), dtype='float32')

        if self.device != 'cpu-gpu':
            # print("self.index.parallel_mode:", self.index.parallel_mode)
            D, I = self.index.search(query, k)
        else:
            # CPU-GPU may result in different result orders compared to CPU-only or GPU-only solution
            # start_ivf = time.time()
            list_IDs = np.empty((nq, nprobe), dtype='int64')
            dist_to_list, list_IDs = self.IVF_index.search(query, nprobe)
            # list_IDs = np.ones((nq, nprobe), dtype='int64')
            # end_ivf = time.time()
            # start_pq = time.time()
            # https://github.com/facebookresearch/faiss/blob/151e3d7be54aec844b6328dc3e7dd0b83fcfa5bc/contrib/ivf_tools.py#L26
            # https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVF.html
            D, I = search_preassigned(self.index, query, k, list_IDs)
            # D, I = replacement_search_preassigned(self.index, query, k, list_IDs, dist_to_list)
            # end_pq = time.time()
            # print("Time for IVF:", end_ivf - start_ivf)
            # print("Time for PQ:", end_pq - start_pq)
        
        out = dict()
        out["id"] = I
        out["dist"] = D

        return out
    
    def retrieve_with_lists(self, query : np.array, list_IDs : np.array, k : Optional[int] = None):
        """
        Search given a list of centroid IDs
        """        

        if k is None: 
            k = self.default_k
    
        nq, dim = query.shape
        assert dim == self.dim
        assert list_IDs.shape[0] == nq
        assert list_IDs.dtype == 'int64'

        D, I = search_preassigned(self.index, query, k, list_IDs)

        out = dict()
        out["id"] = I
        out["dist"] = D

        return out

    def start(self):
        """
        Accept only one connection. Multiple queries per connection.
        """
        communication_socket, address = self.server.accept()
        print("Connection from", address, flush=True)
        
        cnt =  0
        while True:
            print(f"Start query id: {cnt}")
            encoded_queries = b''
            while len(encoded_queries) < self.query_msg_len:
                encoded_queries += communication_socket.recv(self.query_msg_len - len(encoded_queries))

            # print(f"received data from query id: {cnt}")

            if self.request_with_lists:
                k, queries, list_IDs = decode_request_with_lists(encoded_queries, self.batch_size, self.dim, self.nprobe)
                assert queries.shape == (self.batch_size, self.dim)
                assert k == self.default_k
                assert list_IDs.shape == (self.batch_size, self.nprobe)
                out = self.retrieve_with_lists(query=queries, list_IDs=list_IDs, k=k)
            else:
                k, queries = decode_request(encoded_queries, self.batch_size, self.dim)
                assert queries.shape == (self.batch_size, self.dim)
                assert k == self.default_k
                out = self.retrieve(query=queries, k=k)

            D, I = out["dist"], out["id"]
            answer = encode_answer(I, D, k, self.batch_size)

            # print(f"Length of answer is {len(answer)} bytes")
            sent_bytes = 0
            while sent_bytes < len(answer):
                sent_bytes += communication_socket.send(answer[sent_bytes:])
            # print(f"End query id: {cnt}")
            cnt += 1

if __name__ == "__main__":
   
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--dbname", type=str, default="RALM-S1M")
    parser.add_argument("--index_key", type=str, default="IVF32768,PQ32")
    parser.add_argument("--index_dir", type=str, default="/mnt/scratch/wenqi/Faiss_experiments/trained_CPU_indexes/bench_cpu_RALM-S1M_IVF32768,PQ32/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--request_with_lists", type=int, default=0)
    parser.add_argument("--device", type=str, default='gpu')
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--omp_threads", type=int, default=None)


    args = parser.parse_args()
 
    host = args.host
    port = args.port
    batch_size = args.batch_size
    dim = args.dim
    k = args.k
    nprobe = args.nprobe
    request_with_lists = args.request_with_lists
    device = args.device
    ngpu = args.ngpu
    omp_threads = args.omp_threads
    # 128-dim SIFT 
    dbname = args.dbname
    index_key = args.index_key
    index_dir = args.index_dir
    
    faiss_server = FaissServer(
        host=host, port=port,
        index_dir=index_dir, dbname=dbname, index_key=index_key, default_k=k, nprobe=nprobe,
        batch_size=batch_size, dim=dim, request_with_lists=request_with_lists, device=device, ngpu=ngpu, omp_threads=omp_threads)
    faiss_server.start()
