import numpy as np
import torch
from typing import ByteString, Literal, Tuple

BYTE_ORDER_PY: Literal['little', 'big'] = 'big'
BYTE_ORDER_NP: Literal = 'C'

N_BYTES_K: int = 4
N_BYTES_PER_QUERY: int = 4
N_BYTES_PER_IDX: int = 8
N_BYTES_PER_DIST: int = 4

N_BYTES_INT32 : int = 4
N_BYTES_FLOAT32 : int = 4
N_BYTES_AXI : int = 64

def request_message_length(batch_size: int, dim: int):
    return N_BYTES_K + batch_size * (dim * N_BYTES_PER_QUERY)

def request_message_length_with_lists(batch_size: int, dim: int, nprobe: int):
    """ CPU data format """
    return 16 + batch_size * (dim * N_BYTES_FLOAT32 + nprobe * N_BYTES_PER_IDX)

# def request_message_length_with_lists(batch_size: int, dim: int, nprobe: int):
#     """ FPGA data format """
#     n_AXI_header = 1
#     n_AXI_cell_IDs = int(np.ceil(nprobe * N_BYTES_INT32 / N_BYTES_AXI))
#     n_AXI_vector = int(np.ceil(dim * N_BYTES_FLOAT32 / N_BYTES_AXI))
    
#     batch_byte_len = N_BYTES_AXI * n_AXI_header + batch_size * int(N_BYTES_AXI * (n_AXI_cell_IDs + n_AXI_vector * (nprobe + 1)))

#     return batch_byte_len

def answer_message_len(k: int, batch_size: int):
    return batch_size * k * (N_BYTES_PER_IDX + N_BYTES_PER_DIST)


def encode_request(batch_of_queries: np.array, k: int, batch_size: int, dim: int) -> bytes:
    """ Encode a batch of queries and the number of requested nearest neighbours
    as bytes to be sent to a retrieval service.

    Parameters
    ----------
    batch_of_queries : np.array
        size -> (batch size, number hidden dimensions)
    k : int
        number of requested nearest neighbours

    Returns
    -------
    bytes
        a serialized form of the request ready to be sent to the server

    """
    # runtime check if input dimensions correspond to conventions
    assert batch_of_queries.shape == (batch_size, dim)
    # assert batch_of_queries.dtype == torch.float32

    # allocate bytes for the request message
    serialized_request = bytearray(request_message_length(batch_size, dim))

    # write bytes with k and queries
    serialized_request[:N_BYTES_K] = k.to_bytes(4, byteorder=BYTE_ORDER_PY)
    # serialized_request[N_BYTES_K:] = bytearray(batch_of_queries)
    serialized_request[N_BYTES_K:] = batch_of_queries.tobytes(order=BYTE_ORDER_NP)

    return serialized_request

def encode_request_with_lists(batch_of_queries: np.array, list_IDs : np.array, batch_size: int, dim: int, nprobe: int, k: int) -> bytes:
    """
    Format: 
        header: 16 bytes = batch_size, dim, nprobe, k, all in int32
        queries: batch_size * dim * 4 bytes (float32)
        list_IDs: batch_size * nprobe * 8 bytes (int64)
    """
    assert batch_of_queries.shape == (batch_size, dim)
    assert list_IDs.shape == (batch_size, nprobe)

    assert batch_of_queries.dtype == np.float32
    if list_IDs.dtype != np.int64:
        list_IDs = list_IDs.astype(np.int64)
    
    # allocate bytes for the request message
    serialized_request = bytearray(request_message_length_with_lists(batch_size, dim, nprobe))
    
    serialized_request[0:4] = batch_size.to_bytes(4, byteorder=BYTE_ORDER_PY)
    serialized_request[4:8] = dim.to_bytes(4, byteorder=BYTE_ORDER_PY)
    serialized_request[8:12] = nprobe.to_bytes(4, byteorder=BYTE_ORDER_PY)
    serialized_request[12:16] = k.to_bytes(4, byteorder=BYTE_ORDER_PY)
    
    serialized_request[16:16 + batch_size * dim * N_BYTES_FLOAT32] = batch_of_queries.tobytes(order=BYTE_ORDER_NP)
    serialized_request[16 + batch_size * dim * N_BYTES_FLOAT32:16 + batch_size * dim * N_BYTES_FLOAT32 + batch_size * nprobe * N_BYTES_PER_IDX] = list_IDs.tobytes(order=BYTE_ORDER_NP)

    return serialized_request

# def encode_request_with_lists(batch_of_queries: np.array, list_IDs : np.array, list_centroids: np.array, batch_size: int, dim: int, nprobe: int) -> bytes:
#     """
#     Send the query & list centroid information that the FPGA can parse and use for search
    
#     FPGA input (C2F) format:
#     // Format: foe each query
#     // packet 0: header (batch_size, nprobe, terminate)
#     //   for the following packets, for each query
#     // 		packet 1~k: cell_IDs to scan -> size = ceil(nprobe * 4 / 64) 
#     // 		packet k~n: query_vectors
#     // 		packet n~m: center_vectors
#     """
#     n_AXI_header = 1
#     n_AXI_cell_IDs = int(np.ceil(nprobe * N_BYTES_INT32 / N_BYTES_AXI))
#     n_AXI_vector = int(np.ceil(dim * N_BYTES_FLOAT32 / N_BYTES_AXI))
    
#     """
#     Here, send out dummy data (only header valid)
    
#     Dummy retriever (no encoding overhead):
#     Step (overall): total time: 3863.28 ms   average time: 7.55 ms
#     Retrieval: total time: 213.78 ms         average time: 0.42 ms
    
# 	External retriever (dummy data):
# 	Step (overall): total time: 4623.84 ms   average time: 9.03 ms
# 	Retrieval: total time: 832.56 ms         average time: 1.63 ms
    
#     External retriever (cp real data from numpy):
#     Step (overall): total time: 6733.07 ms   average time: 13.15 ms
#     Retrieval: total time: 3221.66 ms        average time: 6.29 ms
#     """

#     # dummy request, without real contents
#     serialized_request = bytearray(request_message_length_with_lists(batch_size, dim, nprobe))
#     headers = np.zeros(3, dtype=np.int32)
#     headers[0] = batch_size
#     headers[1] = nprobe
#     headers[2] = 0 # terminate
#     serialized_request[:3 * 4] = headers.tobytes(order=BYTE_ORDER_NP)

#     # a slow version: convert from np to list
#     # serialized_request = b''
#     # headers = np.zeros(int(n_AXI_header * N_BYTES_AXI / N_BYTES_INT32), dtype=np.int32)
#     # headers[0] = batch_size
#     # headers[1] = nprobe
#     # headers[2] = 0 # terminate
#     # serialized_request += headers.tobytes(order=BYTE_ORDER_NP)
#     # for i in range(batch_size):
        
#     #     cell_IDs = np.zeros(int(n_AXI_cell_IDs * N_BYTES_AXI / N_BYTES_INT32), dtype=np.int32)
#     #     cell_IDs[:nprobe] = list_IDs[i]
#     #     serialized_request += cell_IDs.tobytes(order=BYTE_ORDER_NP)
        
#     #     query_vectors = np.zeros(int(n_AXI_vector * N_BYTES_AXI / N_BYTES_FLOAT32), dtype=np.float32)
#     #     query_vectors[:dim] = batch_of_queries[i]
#     #     serialized_request += query_vectors.tobytes(order=BYTE_ORDER_NP)
        
#     #     center_vectors = np.zeros(int(nprobe * n_AXI_vector * N_BYTES_AXI / N_BYTES_FLOAT32), dtype=np.float32)
#     #     for j in range(nprobe):
#     #         center_vectors[int(j * n_AXI_vector * N_BYTES_AXI / N_BYTES_FLOAT32): int(j * n_AXI_vector *  N_BYTES_AXI / N_BYTES_FLOAT32) + dim] = list_centroids[i][j]
#     #     serialized_request += center_vectors.tobytes(order=BYTE_ORDER_NP)
    
#     # the slowest version: directly convert from np
#     # serialized_request = b''
#     # headers = np.zeros(int(n_AXI_header * N_BYTES_AXI / N_BYTES_INT32), dtype=np.int32)
#     # headers[0] = batch_size
#     # headers[1] = nprobe
#     # headers[2] = 0 # terminate
#     # for i in range(batch_size):
#     #     serialized_request += headers.tobytes(order=BYTE_ORDER_NP)
#     #     serialized_request += list_IDs[i].tobytes(order=BYTE_ORDER_NP)
#     #     serialized_request += batch_of_queries[i].tobytes(order=BYTE_ORDER_NP)
#     #     for j in range(nprobe):
#     #         serialized_request += list_centroids[i][j].tobytes(order=BYTE_ORDER_NP)
    
#     return serialized_request

def decode_request(serialized_request: bytes, batch_size: int, dim: int) -> Tuple[int, np.array]:
    """

    Parameters
    ----------
    serialized_request : bytes
       byte-encoding of a request for the retrieval service. 

    Returns
    -------
    Tuple[int, np.array]
        decoded request
        first element in tuple -> number of nearest neighbours requested
        second element in tuple -> batch of queries with, size -> (batch_size, number_hidden_dimensions)

    """
    # read k from request
    k = int.from_bytes(serialized_request[:N_BYTES_K], byteorder=BYTE_ORDER_PY)

    # read queries as numpy array
    queries = np.frombuffer(serialized_request[N_BYTES_K:], dtype=np.float32)
    queries = queries.reshape((batch_size, dim))

    return k, queries

def decode_request_with_lists(serialized_request: bytes, batch_size: int, dim: int, nprobe: int) -> Tuple[int, np.array, np.array]:
    """
    Format: 
        header: 16 bytes = batch_size, dim, nprobe, k, all in int32
        queries: batch_size * dim * 4 bytes (float32)
        list_IDs: batch_size * nprobe * 8 bytes (int64)

    return k, queries, list_IDs
    """
    decoded_batch_size = int.from_bytes(serialized_request[:4], byteorder=BYTE_ORDER_PY)
    decoded_dim = int.from_bytes(serialized_request[4:8], byteorder=BYTE_ORDER_PY)
    decoded_nprobe = int.from_bytes(serialized_request[8:12], byteorder=BYTE_ORDER_PY)
    decoded_k = int.from_bytes(serialized_request[12:16], byteorder=BYTE_ORDER_PY)
    assert decoded_batch_size == batch_size
    assert decoded_dim == dim
    assert decoded_nprobe == nprobe

    k = decoded_k
    queries = np.frombuffer(serialized_request[16:16 + batch_size * dim * N_BYTES_FLOAT32], dtype=np.float32)
    queries = queries.reshape((batch_size, dim))
    list_IDs = np.frombuffer(serialized_request[16 + batch_size * dim * N_BYTES_FLOAT32:16 + batch_size * dim * N_BYTES_FLOAT32 + batch_size * nprobe * N_BYTES_PER_IDX], dtype=np.int64)
    list_IDs = list_IDs.reshape((batch_size, nprobe))

    return k, queries, list_IDs

def encode_answer(indices: np.array, distances: np.array, k: int, batch_size: int) -> bytes:
    """

    Parameters
    ----------
    indices : np.array
        vector with indices (int64) for the batch return by retrieval service

    distances : np.array
        vector with distances (float32) for each neighbour

    k : int
        number of requested neighbours helps to create data-structure and check dimensions


    Returns
    -------
    bytes
        a serialized form of the answer ready to be sent back back to the client

    """
    assert indices.shape == (batch_size, k)
    assert indices.dtype == np.int64
    assert distances.shape == (batch_size, k)
    assert distances.dtype == np.float32

    # allocate bytes for the answer message
    serialized_answer = bytearray(answer_message_len(k, batch_size))

    # write bytes with indices and distances
    distances_fist_byte = batch_size * k * N_BYTES_PER_IDX
    serialized_answer[:distances_fist_byte] = bytearray(indices)
    serialized_answer[distances_fist_byte:] = bytearray(distances)

    return serialized_answer


def decode_answer(serialized_answer: bytes, k: int, batch_size: int) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    serialized_answer: bytes :
       byte-encoding of the answer to a request to a retrieval service. 

    k: int :
        number of requested neighbours


    Returns
    -------
    Tuple[np.array, np.array]
        decoded answer
        first element in tuple -> batch of indices, size -> (batch_size, k)
        second element in tuple -> batch of distances of neighbours to batch of queries, size -> (batch_size, k)

    """
    # separate indices and distances from input bytes
    distances_first_byte = batch_size * k * N_BYTES_PER_IDX
    indices_bytes = serialized_answer[:distances_first_byte]
    distances_bytes = serialized_answer[distances_first_byte:]

    # read indices and distances as numpy arrays
    indices = np.frombuffer(indices_bytes, dtype=np.int64)
    distances = np.frombuffer(distances_bytes, dtype=np.float32)

    indices = indices.reshape((batch_size, k))
    distances = distances.reshape((batch_size, k))

    return indices, distances
