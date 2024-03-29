import os
import pickle
import numpy as np

def save_obj(obj, dirc, name):
    # note use "dir/" in dirc
    with open(os.path.join(dirc, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, protocol=4) # for py37,pickle.HIGHEST_PROTOCOL=4

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def parse_arch(input='8CPU-1GPU'):
    """
    return device, omp_threads, ngpu
    """
    if 'CPU' in input and 'GPU' in input:
        device = 'CPU-GPU'
        split_input = input.split('-')
        omp_threads = int(split_input[0].split('CPU')[0])
        ngpu = int(split_input[1].split('GPU')[0])
    elif 'CPU' in input and not 'GPU' in input:
        device = 'CPU'
        omp_threads = int(input.split('CPU')[0])
        ngpu = None
    elif 'GPU' in input and not 'CPU' in input:
        device = 'GPU'
        omp_threads = None
        ngpu = int(input.split('GPU')[0])
    else:
        raise ValueError('Invalid architecture input')
    
    return device, omp_threads, ngpu

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def read_deep_fbin(filename):
    """
    Read *.fbin file that contains float32 vectors

    All embedding data is stored in .fbin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (float32)]

    The groundtruth is stored in .ibin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (int32)]

    https://research.yandex.com/datasets/biganns
    https://pastebin.com/BAf6bM5L
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)

    arr = np.memmap(filename, dtype=np.float32, offset=8, mode='r')
    return arr.reshape(nvecs, dim)

def generate_queries(dbname, batch_size, n_batches, base_dir='/mnt/scratch/wenqi/Faiss_experiments/'):
    """
    Return a list of query batches, each is a numpy array
    """
    pass

    if dbname.startswith('SIFT'):
        xq = mmap_bvecs(os.path.join(base_dir, 'bigann/bigann_query.bvecs'))
        xq = xq.astype('float32').copy()
    elif dbname.startswith('Deep'):
        xq = read_deep_fbin(os.path.join(base_dir, 'deep1b/query.public.10K.fbin'))
        xq = xq.astype('float32').copy()
    elif dbname.startswith('RALM'):
        if dbname.startswith('RALM-S'):
            dim_replicate_factor = 4 # dim = 512
        elif dbname.startswith('RALM-L'):
            dim_replicate_factor = 8 # dim = 1024
        else:
            raise ValueError('Unknown RALM dataset')
        xq = mmap_bvecs(os.path.join(base_dir, 'bigann/bigann_query.bvecs'))
        xq = xq.astype('float32').copy()
        xq = np.tile(xq, (1, dim_replicate_factor))

    if batch_size > xq.shape[0]:
        xq = np.tile(xq, (int(np.ceil(batch_size/nq)), 1))
    nq, dim = xq.shape

    # For RALM-L, reverse the query order, such that the query order is different to RALM-S
    if dbname.startswith('RALM-L'):
        xq = xq[::-1]
    xq = np.ascontiguousarray(xq)

    query_list_one_pass = []
    for i in range(int(nq / batch_size)):
        query_list_one_pass.append(xq[i*batch_size:(i+1)*batch_size])
        if i > n_batches:
            break

    query_list = []
    for i in range(n_batches):
        query_list.append(query_list_one_pass[i % len(query_list_one_pass)])

    assert len(query_list) == n_batches

    return query_list