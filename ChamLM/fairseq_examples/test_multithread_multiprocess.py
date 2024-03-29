"""
This script is used to test the speed of multithread and multiprocess, by using a simple function my_func(x) = x * x

Conclusion:
    * the threading library is much faster than multiprocessing library
    * multiprocess is much slower than multithread
    
(pytorch) wejiang@sgs-gpu02:/mnt/scratch/wenqi/knn-transformer/fairseq_examples$ python test_multiprocess.py 
Threading: 0.00033402442932128906 -> 0.3 ms
Multiprocesssing thread pool: 0.0009152889251708984 -> 0.9 ms
Process pool: 0.33405470848083496 -> 334 ms
"""


from time import time
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import threading
import numpy as np
import pickle

def my_func(x):
    return x * x

def main():
    arr = np.ones((1024, 1024, 1024), dtype=np.uint8)
    expected_sum = np.sum(arr)

    x = threading.Thread(target=my_func, args=(1,))
    start = time()
    x.start()
    x.join()
    print("Threading:", time() - start)

    with ThreadPool(1) as threadpool:
        start = time()
        # assert (
        #     threadpool.apply(np.sum, (arr,)) == expected_sum
        # )
        resultAsync = threadpool.apply_async(my_func, (1,))
        resultAsync.get()
        print("Multiprocesssing thread pool:", time() - start)

    with mp.Pool(1) as p:
        start = time()
        print(p.map(my_func, [1, 2, 3]))
        print("Process pool (implementation 1): ", time() - start)


    p_list = []
    num_proc = 1
    for i in range(num_proc):
        p_list.append(mp.Process(target=my_func, args=(i + 1,)))

    start = time()
    for i in range(num_proc):
        p_list[i].start()
    for i in range(num_proc):
        p_list[i].join()
    print("Process pool (implementation 2): ", time() - start)


    with mp.get_context("spawn").Pool(1) as processpool:
        start = time()
        # assert (
        #     processpool.apply(np.sum, (arr,)) == expected_sum
        # )
        processpool.apply(my_func, (1,))
        print("Process pool (implementation 3):", time() - start)

if __name__ == "__main__":
    main()