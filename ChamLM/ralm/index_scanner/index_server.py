"""
This is an index server processes used in a pure vector DB (not in a RALM).

The index server scans the index and send a list of IDs to the PQ scan server, either a Faiss server or an FPGA coordinator process.

Example usage:
    python index_server.py --host "127.0.0.1" --port 9090 --dim 512 --nlist 32768 --nprobe 32 --k 10 --batch_size 32 --n_batch 1 --device "gpu" --use_gpu_id 0 --omp_threads 8 --use_tiktok 0
"""

import time
import numpy as np

from typing import Optional, List

from ralm.retriever.retriever import ExternalRetriever
from ralm.index_scanner.index_scanner import IndexScanner

class IndexServer:
    
    def __init__(self, 
                host: Optional[str] = None, port: Optional[int] = 9090,
                dim : Optional[int] = 1024, 
                nlist : Optional[int] = 32768, 
                nprobe : Optional[int] = 32, 
                default_k : Optional[int] = 10,
                batch_size : Optional[int] = 1,
				centroids : Optional[np.array] = None,
                device : Optional[str] = 'gpu', # 'cpu' or 'gpu' 
                use_gpu_id : Optional[int] = None,
                omp_threads : Optional[int] = None):
        
        self.host = host
        self.port = port
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.default_k = default_k
        self.batch_size = batch_size
        self.device = device
        self.use_gpu_id = use_gpu_id
        self.omp_threads = omp_threads

        self.retriever = ExternalRetriever(
            host=host, port=port, batch_size=batch_size, dim=dim, default_k=default_k)
        
        self.index_scanner = IndexScanner(
            dim=dim, nlist=nlist, nprobe=nprobe, centroids=centroids, device=device, use_gpu_id=use_gpu_id, omp_threads=omp_threads)
        print("Finish initializing the index server.")

    def search_single_batch_send(self, queries: np.array, nprobe : Optional[int] = None, k : Optional[int] = None):
        """
        Search the IVF idnex and send out a single query

        Query dimension (np array): (batch_size, dim)
        """        
        assert queries.shape == (self.batch_size, self.dim)
    
        if nprobe is None:
            nprobe = self.nprobe
        if k is None:
            k = self.default_k
        
        list_IDs, list_centroids = self.index_scanner.search(queries, nprobe)
        self.retriever.retrieve_with_lists_send(queries, list_IDs, k)

    def search_single_batch_recv(self, k : Optional[int] = None):
        if k is None:
            k = self.default_k
        indices, distances = self.retriever.retrieve_recv(k)
        assert indices.shape == (self.batch_size, k)
        return indices, distances

    def search_single_batch(self, queries: np.array, nprobe : Optional[int] = None, k : Optional[int] = None):
        """
        Query dimension (np array): (batch_size, dim)

        Return: ANN results including indices, distances (shape = (batch_size, k)
        """
        self.search_single_batch_send(queries, nprobe, k)
        indices, distances = self.search_single_batch_recv(k)

        return indices, distances

    def search_multi_batch(self, query_batch_list: List[np.array], nprobe : Optional[int] = None, k : Optional[int] = None):
        """
        query_batch_list: a list of queries, each with dimension (np array): (batch_size, dim)

        Return: ANN results including indices, distances (shape = (n_batch, batch_size, k)
        """
        
        indices = []
        distances = []
        self.latency_list_sec = []

        t_start = time.time()
        for query_batch in query_batch_list:
            t_start_batch = time.time()
            batch_indices, batch_distances = self.search_single_batch(query_batch, nprobe, k)
            indices.append(batch_indices)
            distances.append(batch_distances)
            t_end_batch = time.time()
            self.latency_list_sec.append(t_end_batch - t_start_batch)

        indices = np.array(indices, dtype=np.int64)
        distances = np.array(distances, dtype=np.float32)

        t_end = time.time()
        self.throughput_batches_per_sec = len(query_batch_list) / (t_end - t_start)
        
        return indices, distances


    def search_multi_batch_tiktok(self, query_batch_list: List[np.array], nprobe : Optional[int] = None, k : Optional[int] = None):
        """
        High-throughput multi-batch search

        Input: 
          query_batch_list: a list of queries, each with dimension (np array): (batch_size, dim)

        Return: ANN results including indices, distances (shape = (n_batch, batch_size, k)
        """

        indices = []
        distances = []

        req_on_fly = [] # 'tik' or 'tok', at most both on fly
        total_batches = len(query_batch_list)
        batch_id_to_run = 0
        finished_batches = 0

        t_start_batch = {"tik": [], "tok": []}
        t_end_batch = {"tik": [], "tok": []}

        t_start = time.time()
        while finished_batches < total_batches:

            for step_type in ['tik', 'tok']:

                # not send request yet
                if step_type not in req_on_fly:
                    if batch_id_to_run < total_batches:
                        t_start_batch[step_type].append(time.time())
                        print("Current batch id: {}".format(batch_id_to_run))
                        self.search_single_batch_send(query_batch_list[batch_id_to_run], nprobe, k)
                        req_on_fly.append(step_type)
                        batch_id_to_run += 1
                # send request but not received yet
                else:
                    # tik is the first request to be received and there is data in the reply buffer
                    if req_on_fly[0] == step_type:
                        if self.retriever.poll():
                            batch_indices, batch_distances = self.search_single_batch_recv(k)
                            indices.append(batch_indices)
                            distances.append(batch_distances)
                            req_on_fly.remove(step_type)
                            finished_batches += 1
                            t_end_batch[step_type].append(time.time())
        
        indices = np.array(indices, dtype=np.int64)
        distances = np.array(distances, dtype=np.float32)
       
        t_end = time.time()
        self.throughput_batches_per_sec = total_batches / (t_end - t_start) 

        self.latency_list_sec = []
        for i in range(int(total_batches/2)):
            self.latency_list_sec.append(t_end_batch['tik'][i] - t_start_batch['tik'][i])
            self.latency_list_sec.append(t_end_batch['tok'][i] - t_start_batch['tok'][i])
        if total_batches % 2 == 1:
            self.latency_list_sec.append(t_end_batch['tik'][-1] - t_start_batch['tik'][-1])
        assert len(self.latency_list_sec) == total_batches

        return indices, distances

    def get_profiling(self):
        """
        Can be called after search_multi_batch / search_multi_batch_tiktok

        Return: latency_list (sec), throughput (batches / sec)
        """
        latency_list_sec = np.array(self.latency_list_sec)
        throughput_batches_per_sec = self.throughput_batches_per_sec

        return latency_list_sec, throughput_batches_per_sec
        
if __name__ == '__main__':

    import argparse 
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--nlist', type=int, default=32768)
    parser.add_argument('--nprobe', type=int, default=32)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_batch', type=int, default=1, help="number of queries to send to the server")
    parser.add_argument('--device', type=str, default='gpu', help="cpu or gpu")
    parser.add_argument('--use_gpu_id', type=int, default=None, help="the GPU ID to run the model")
    parser.add_argument('--omp_threads', type=int, default=None, help="the number of OpenMP threads")
    parser.add_argument('--use_tiktok', type=int, default=0, help="0 or 1, whether to use tiktok")

    args = parser.parse_args()

    host = args.host
    port = args.port
    dim = args.dim
    nlist = args.nlist
    nprobe = args.nprobe
    k = args.k
    batch_size = args.batch_size
    n_batch = args.n_batch
    device = args.device
    use_gpu_id = args.use_gpu_id
    omp_threads = args.omp_threads
    use_tiktok = args.use_tiktok

    index_server = IndexServer(
        host=host, port=port, dim=dim, nlist=nlist, nprobe=nprobe, default_k=k, batch_size=batch_size,
        device=device, use_gpu_id=use_gpu_id, omp_threads=omp_threads)
    
    query_list = []
    for i in range(n_batch):
        query_list.append(np.random.rand(batch_size, dim).astype(np.float32))

    t_start = time.time()
    if use_tiktok:
        indices, distances = index_server.search_multi_batch_tiktok(query_list, nprobe, k)
    else:
        indices, distances = index_server.search_multi_batch(query_list, nprobe, k)
    t_end = time.time()

    latency_list_sec, throughput_batches_per_sec = index_server.get_profiling()

    print("Batch size: {}".format(batch_size))
    print("Total time: {:.2f} ms".format((t_end - t_start) * 1000))
    print("Throughput (steps * batch_size / sec): {:.2f}".format(
        batch_size * throughput_batches_per_sec))