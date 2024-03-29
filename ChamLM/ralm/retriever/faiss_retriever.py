"""
The RALM's local Faiss retriever.

Currently, no ground truth evaluation is supported. 
"""
import torch
import time
import os 
import faiss
import numpy as np

from typing import Optional, List
from ralm.retriever.retriever import BaseRetriever
from faiss.contrib.ivf_tools import search_preassigned
from faiss import swig_ptr


class LocalFaissRetriever(BaseRetriever):

    def __init__(self, index_dir, dbname, index_key, default_k : Optional[int] = 10, nprobe : Optional[int] = 1,
                device : Optional[str] = 'cpu', # 'cpu' or 'gpu' or 'cpu-gpu'
                # ngpu, use_float16, use_precomputed_tables are used only when device == 'gpu 
                ngpu : Optional[int] = 1, 
                start_gpu_id : Optional[int] = 0,
                omp_threads : Optional[int] = None,
                use_float16 : Optional[bool] = True, 
                use_precomputed_tables : Optional[bool] = True
                ):
        """
        index_dir: the directory storing a trained and populated faiss index, e.g.,
            /mnt/scratch/wenqi/Faiss_experiments/trained_CPU_indexes/bench_cpu_SIFT1000M_IVF16384,PQ16
        dbname: the name of the dataset, e.g., SIFT1000M
        index_key: the index name of a faiss index, e.g., IVF16384,PQ16 or SBERT1M_IVF4096,Flat
        default_k: the default k to return during a search, can be overwritten in the retrieve function

        device: 'cpu' or 'gpu' or 'cpu-gpu'
            'cpu': both IVF and PQ on cpu
            'gpu': both IVF and PQ on gpu	
            'cpu-gpu': IVF on gpu, PQ on cpu
        """

        self.index_dir = index_dir
        self.dbname = dbname
        self.index_key = index_key 
        self.default_k = default_k
        self.nprobe = nprobe
        
        self.index = self.get_populated_index()
        self.dim = self.index.d
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)
        self.set_nprobe(nprobe)
        self.device = device
        self.ngpu = ngpu
        self.start_gpu_id = start_gpu_id
        
        if self.device == 'gpu':
            self.entire_index_to_gpu(ngpu=ngpu, start_gpu_id=start_gpu_id)
        elif self.device == 'cpu-gpu':
            self.IVF_index_to_gpu(ngpu=ngpu, start_gpu_id=start_gpu_id)
            """
            search_preassigned must be set with a proper parallel mode:
                if call search itself, the parallel for will be outside the search_preassigned function
                    https://github.com/facebookresearch/faiss/blob/v1.7.2/faiss/IndexIVF.cpp#L350
                if parallel_mode = 0 (default), the do_parallel will be set as false (single thread):
                    https://github.com/facebookresearch/faiss/blob/v1.7.2/faiss/IndexIVF.cpp#L416

            mode 3 is the fastest mode in the test
            """
            self.index.parallel_mode = 3

        # 'cpu' or 'cpu-gpu'
        if omp_threads is not None:
            print("WARNING: setting omp thread number to", omp_threads, 
                  ", please make sure only one Faiss object exists in the current process, "
                  "otherwise it can affect the performance of other Faiss objects.")
            self.omp_threads = omp_threads
            faiss.omp_set_num_threads(self.omp_threads)   

        # warm up search
        query = np.random.rand(1, self.dim).astype('float32')
        self.retrieve(query, nprobe=1, k=1)

    def get_populated_index(self):
        """
        Load trained faiss index.
        """
        filename = os.path.join(self.index_dir, 
              f"{self.dbname}_{self.index_key}_populated.index")
        print("loading", filename)
        index = faiss.read_index(filename)

        return index

    def entire_index_to_gpu(self, ngpu : Optional[int] = 1, 
        start_gpu_id : Optional[int] = 0,
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
        
        print("Moving the entire index to the GPU...")
        gpus = [i + start_gpu_id for i in range(ngpu)]
        self.index = faiss.index_cpu_to_gpus_list(index=self.index, gpus=gpus)

        """
        The older faiss interface does not work anymore... segment fault during cpu to gpu
        vres, vdev = self.make_vres_vdev(ngpu=ngpu, start_gpu=start_gpu, tempmem=tempmem)
        # print(vres, vdev)
        # self.index = faiss.index_cpu_to_gpu_multiple(
        # 	vres, vdev, self.index, co)
        """
        print("Index to gpu finished.")

    def IVF_index_to_gpu(self, ngpu : Optional[int] = 1, start_gpu_id : Optional[int] = 0):
        """
        Only copy the IVF index (in IVF-PQ) to the GPU
        """
        print("Moving only the IVF index to the GPU...")
        self.IVF_index = faiss.downcast_index(self.index.quantizer)
        gpus = [i + start_gpu_id for i in range(ngpu)]
        self.IVF_index = faiss.index_cpu_to_gpus_list(index=self.IVF_index, gpus=gpus)
        print("Index to gpu finished.")


    def make_vres_vdev(self, ngpu : Optional[int] = 1, start_gpu : Optional[int] = 0, tempmem : Optional[int] = -1):
        """
        DEPRECATED due to the cpu to gpu API change in Faiss
        return vectors of device ids and resources useful for gpu_multiple
        
        Input args:
            start_gpu: the first GPU ID working on faiss, one can choose to start with 1 if 0 is in use
            ngpu: the total number of GPUs used for retrieval
            tempmem = -1  # if -1, use system default
        """
        gpu_resources = []
        for i in range(faiss.get_num_gpus()):
            res = faiss.StandardGpuResources()
            if tempmem >= 0:
                res.setTempMemory(tempmem)
            gpu_resources.append(res)

        print("1")	
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
    
        print("2")		
        for i in range(start_gpu, start_gpu + ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
            print("i:",i)	
        return vres, vdev

    def set_nprobe(self, nprobe : int):

        self.nprobe = nprobe
        self.index.nprobe = nprobe
        # param = "nprobe={}".format(nprobe*1.0)
        # self.ps.set_index_parameters(self.index, 'nprobe', nprobe * 1.0)
        # self.ps.set_index_parameters(self.index, param)

    def replacement_search_preassigned(self, index, x, k, Iq, Dq):
        """Find the k nearest neighbors of the set of vectors x in an IVF index,
        with precalculated coarse quantization assignment.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        k : int
            Number of nearest neighbors.
        Dq : array_like, optional
            Distance array to the centroids, size (n, nprobe)
        Iq : array_like, optional
            Nearest centroids, size (n, nprobe)

        Returns
        -------
        D : array_like
            Distances of the nearest neighbors, shape (n, k). When not enough results are found
            the label is set to +Inf or -Inf.
        I : array_like
            Labels of the nearest neighbors, shape (n, k).
            When not enough results are found, the label is set to -1
        """
        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')

        D = np.empty((n, k), dtype=np.float32)
        assert D.shape == (n, k)
        I = np.empty((n, k), dtype=np.int64)

        Iq = np.ascontiguousarray(Iq, dtype='int64')
        assert Iq.shape == (n, index.nprobe)

        if Dq is not None:
            Dq = np.ascontiguousarray(Dq, dtype='float32')
            assert Dq.shape == Iq.shape

        index.search_preassigned(
        # index.search_preassigned_c(
            n, swig_ptr(x),
            k,
            swig_ptr(Iq), swig_ptr(Dq),
            swig_ptr(D), swig_ptr(I),
            False
        )
        return D, I

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
            # D, I = self.replacement_search_preassigned(self.index, query, k, list_IDs, dist_to_list)
            # end_pq = time.time()
            # print("Time for IVF:", end_ivf - start_ivf)
            # print("Time for PQ:", end_pq - start_pq)
        
        out = dict()
        out["id"] = I
        out["dist"] = D

        return out
    
if __name__ == '__main__':
    
    dbname = "SIFT100M"
    index_key = "IVF32768,PQ16"
    index_dir = f"/mnt/scratch/wenqi/Faiss_experiments/trained_CPU_indexes/bench_cpu_{dbname}_{index_key}"
    k = 10
    nprobe = 32
    device = 'cpu'
    # device = 'gpu'
    # device = 'cpu-gpu'
    ngpu = 1
    nq = 10000
    dim = 128
    omp_threads = None # 4

    query = np.random.rand(nq, dim).astype('float32')

    retriever_cpu = LocalFaissRetriever(index_dir=index_dir, dbname=dbname, index_key=index_key, default_k=k, nprobe=nprobe, device=device, ngpu=ngpu, omp_threads=omp_threads)
    start_cpu = time.time()
    out_cpu = retriever_cpu.retrieve(query=query, nprobe=nprobe, k=k)
    end_cpu = time.time()
    
    device = 'gpu'
    retriever_gpu = LocalFaissRetriever(index_dir=index_dir, dbname=dbname, index_key=index_key, default_k=k, nprobe=nprobe, device=device, ngpu=ngpu, omp_threads=omp_threads)
    start_gpu = time.time()
    out_gpu = retriever_gpu.retrieve(query=query, nprobe=nprobe, k=k)
    end_gpu = time.time()
    # assert np.array_equal(out_cpu['id'], out_gpu['id'])

    device = 'cpu-gpu'
    retriever_cpu_gpu = LocalFaissRetriever(index_dir=index_dir, dbname=dbname, index_key=index_key, default_k=k, nprobe=nprobe, device=device, ngpu=ngpu, omp_threads=omp_threads)
    start_cpu_gpu = time.time()
    out_cpu_gpu = retriever_cpu_gpu.retrieve(query=query, nprobe=nprobe, k=k)
    end_cpu_gpu = time.time()

    # assert np.sum(out_cpu['id']) == np.sum(out_cpu_gpu['id'])

    print("CPU time:", end_cpu - start_cpu)
    print("GPU time:", end_gpu - start_gpu)
    print("CPU-GPU time:", end_cpu_gpu - start_cpu_gpu)

    print('CPU results:\t', sorted(out_cpu['id'][0]))
    print('GPU results:\t', sorted(out_gpu['id'][0]))
    print('CPU-GPU results:\t', sorted(out_cpu_gpu['id'][0]))
    print('Seems that CPU-only, GPU-only CPU-GPU results may differ. It could because the pre-assigned list are searched in different orders.')