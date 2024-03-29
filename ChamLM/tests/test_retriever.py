import numpy as np
import torch

from ralm.retriever.serialization_utils import encode_request, encode_request_with_lists, decode_request, decode_request_with_lists, encode_answer, decode_answer, answer_message_len
from ralm.server.server import RandomAnswerServer
from ralm.server.faiss_server import FaissServer
from ralm.retriever.retriever import ExternalRetriever
from ralm.coordinator.retriever_coordinator_server import RetrieveCoordinator
from threading import Thread
import time

base_port = 9090  # +1 for each test needing this base port


def test_serialization_utils():

    # Client side:
    batch_size = 32
    dim = 512
    k_original = 2
    nprobe = 10

    original_queries = np.random.rand(batch_size, dim).astype('float32')
    list_IDs = np.random.randint(0, 100, size=(batch_size, nprobe), dtype=np.int64)
    encoded_request = encode_request(original_queries, k_original, batch_size, dim)
    encoded_request_with_lists = encode_request_with_lists(original_queries, list_IDs, batch_size, dim, nprobe, k_original)
    
    # server side:
    decoded_k, decoded_queries = decode_request(encoded_request, batch_size, dim)
    decoded_with_lists_k, decoded_with_lists_queries, decoded_with_lists_list_IDs = decode_request_with_lists(encoded_request_with_lists, batch_size, dim, nprobe)

    original_indices = np.array(list(range(batch_size * decoded_k)), dtype=np.int64)
    original_indices = original_indices.reshape(batch_size, decoded_k)
    original_distances = np.random.default_rng().standard_normal(size=(batch_size, decoded_k), dtype='float32')
    print(decoded_k, batch_size)
    serialized_answer = encode_answer(original_indices, original_distances, decoded_k, batch_size)

    # Client side:
    decoded_indices, decoded_distances = decode_answer(serialized_answer, k_original, batch_size)
    print(original_indices)
    print(decoded_indices)

    assert k_original == decoded_k
    assert np.array_equal(original_queries, decoded_queries)
    assert np.array_equal(original_indices, decoded_indices)
    assert np.array_equal(original_distances, decoded_distances)
    
    assert k_original == decoded_with_lists_k
    assert np.array_equal(original_queries, decoded_with_lists_queries)
    assert np.array_equal(list_IDs, decoded_with_lists_list_IDs)


# def test_external_retriever():
#     """
#     Caution: There is an issue with this test.
#     The server is started in a daemonized thread.
#     If the server fails for some reason the error message is not properly shown.
#     Further there is a delay from when the process is terminated until the socket is released.
#     This leads to the test failing (OSError) when invoked a second time within a short period.
#     A cooldown of 20 seconds seems to be sufficient.
#     """
#     # Testing solution with mock-server as described here:
#     # https://realpython.com/testing-third-party-apis-with-mock-servers/

#     # Start running mock server in a separate thread.
#     # Daemon threads automatically shut down when the main process exits.

#     batch_size = 32
#     dim = 512
#     k = 2

#     global base_port
#     base_port += 1

#     server = RandomAnswerServer(port=base_port, batch_size=batch_size, dim=dim)
#     mock_server_thread = Thread(target=server.start_one_query_per_conn)
#     mock_server_thread.daemon = True
#     mock_server_thread.start()

#     retriever = ExternalRetriever(port=base_port, batch_size=batch_size, dim=dim)

#     # create and send some test data to the server
#     # query = torch.rand(batch_size, dim, dtype=torch.float32)
#     query = np.random.rand(batch_size, dim).astype('float32')

#     indices, distances = retriever.retrieve(query, k)


# # def test_multi_retriever_coordinator():
# #     """
# #     Test the coordination between the multiple GPU retrieval processes (ralm/retriever/retriever.py), 
# #         the aggregator / cooridnator (ralm/retriever/retriever_coordinator_server.py), 
# #         and the CPU index server (ralm/server/server.py)
# #     """

# #     batch_size = 64
# #     dim = 1024
# #     k = 10
# #     ngpus = 2
# #     num_queries_per_gpu = 10

# #     index_server_port = 9091
# #     index_server_host = 'localhost'

# #     coordinator_port = 9090
# #     coordinator_host = 'localhost'

# #     # CPU index server
# #     cpu_index_server = RandomAnswerServer(
# #         host=index_server_host, port=index_server_port,
# #         batch_size=batch_size, dim=dim)
# #     mock_server_thread = Thread(target=cpu_index_server.start)
# #     mock_server_thread.daemon = True
# #     mock_server_thread.start()

# #     # Coordinator server
            
# #     coordinator = RetrieveCoordinator(
# #         index_server_host=index_server_host, index_server_port=index_server_port,
# #         local_host=coordinator_host, local_port=coordinator_port,
# #         ngpus=ngpus, batch_size=batch_size, num_queries_per_gpu=num_queries_per_gpu, 
# #         k=k, dim=dim)
# #     coordinator_thread = Thread(target=coordinator.start)
# #     coordinator_thread.daemon = True
# #     coordinator_thread.start()
    
# #     # Retrieval processes
# #     retrievers = []
# #     query = torch.rand(batch_size, dim, dtype=torch.float32)
# #     for i in range(ngpus):
# #         retrievers.append(ExternalRetriever(
# #             host=coordinator_host, port=coordinator_port,
# #             batch_size=batch_size, dim=dim))
        
# #     for _ in range(num_queries_per_gpu):
# #         for i in range(ngpus):
# #             retriever_thread = Thread(target=retrievers[i].retrieve, args=(query, k))
# #             retriever_thread.daemon = True
# #             retriever_thread.start()

# #     # TODO: join threads...
    
# def test_external_faiss_retriever():
#     """
#     Caution: There is an issue with this test.
#     The server is started in a daemonized thread.
#     If the server fails for some reason the error message is not properly shown.
#     Further there is a delay from when the process is terminated until the socket is released.
#     This leads to the test failing (OSError) when invoked a second time within a short period.
#     A cooldown of 20 seconds seems to be sufficient.
#     """
#     # Testing solution with mock-server as described here:
#     # https://realpython.com/testing-third-party-apis-with-mock-servers/

#     # Start running mock server in a separate thread.
#     # Daemon threads automatically shut down when the main process exits.

#     dbname = "SIFT1M"
#     index_key = "IVF4096,PQ16"
#     index_dir = f"/mnt/scratch/wenqi/Faiss_experiments/trained_CPU_indexes/bench_cpu_{dbname}_{index_key}"
#     batch_size = 32
#     dim = 128
#     k = 10
#     device = 'gpu'

#     global base_port
#     base_port += 1

#     server = FaissServer(port=base_port, dbname=dbname, index_key=index_key, index_dir=index_dir, batch_size=batch_size, dim=dim, device=device)
#     mock_server_thread = Thread(target=server.start)
#     mock_server_thread.daemon = True
#     mock_server_thread.start()

#     retriever = ExternalRetriever(port=base_port, batch_size=batch_size, dim=dim)

#     # create and send some test data to the server
#     # query = torch.rand(batch_size, dim, dtype=torch.float32)
#     query = np.random.rand(batch_size, dim).astype('float32')

#     indices, distances = retriever.retrieve(query, k)
