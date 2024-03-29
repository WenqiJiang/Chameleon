"""
A local IVF index scanner on CPU/GPU.

It can be used in two ways:
1. combined with RALM to perform retrieval on a local index and request with list IDs.
2. used with a retriever to serve as the index server, sending requests to either (a)
    an FPGA coordinator process or (b) a Faiss server supporting request with list IDs.
"""

import faiss
import time
import numpy as np

from typing import Optional, List

class IndexScanner:

    def __init__(self, dim : Optional[int] = 1024, nlist : Optional[int] = 32768, 
                 nprobe : Optional[int] = 32, 
                 centroids : Optional[np.array] = None,
                 device : Optional[str] = 'gpu', # 'cpu' or 'gpu' 
                 use_gpu_id : Optional[int] = None,
                 omp_threads : Optional[int] = None):
        """
        Instantiate the GPU index and warm up the search
        """
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe

        # create index
        if centroids is None:
            print("No IVF index provided, creating IVF index...")
            self.index = faiss.IndexFlatL2(self.dim)
            self.centroids = np.random.rand(self.nlist, self.dim).astype('float32')
            self.index.add(self.centroids)
        else:
            print("Loading IVF index...")
            self.index = faiss.IndexFlatL2(self.dim)
            self.centroids = centroids
            assert self.centroids.shape == (self.nlist, self.dim) 
            self.index.add(self.centroids)
        
        if device == 'gpu':
            use_gpu_id = use_gpu_id if use_gpu_id is not None else 0
            self.resource = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.resource, use_gpu_id, self.index) # 1 as ID

        if omp_threads is not None and device != 'gpu':
            print("WARNING: setting omp thread number to", omp_threads, 
                  ", please make sure only one Faiss object exists in the current process, "
                  "otherwise it can affect the performance of other Faiss objects.")
            self.omp_threads = omp_threads
            faiss.omp_set_num_threads(self.omp_threads)   
            
        # warm up
        random_query = np.random.rand(1, dim).astype('float32')
        self.index.search(random_query, nprobe)
        print("Finish initializing the index scanner.")

    def search(self, queries: np.array, nprobe : Optional[int] = None):
        """
        Query dimension (np array): (batch_size, dim)

        Return: list IDs and respective centroid vectors
        """
        assert queries.shape[1] == self.dim 
        nq = queries.shape[0]
        if nprobe is None:
            nprobe = self.nprobe

        # D, I Shape = nq, nprobe
        D, I = self.index.search(queries, self.nprobe)
        list_IDs = np.array(I, dtype='int64')
        list_centroids = self.centroids[list_IDs.flatten()].reshape(nq, nprobe, self.dim)

        return list_IDs, list_centroids
        

if __name__ == "__main__":

    dim = 128
    nlist = 32768
    nprobe = 32
    use_gpu_id = 0

    index_scanner = IndexScanner(dim=dim, nlist=nlist, nprobe=nprobe, use_gpu_id=use_gpu_id)
    
    nq = 256
    queries = np.random.rand(nq, dim).astype('float32')

    start = time.time()
    list_IDs, list_centroids = index_scanner.search(queries, nprobe=nprobe)
    end = time.time()
    print("Time: {:.2f} ms".format((end - start) * 1000))
    print("QPS:", nq / (end - start))