"""
The RALM's retriever interface to the retriever service.

Example Usage:
    cd ralm/retriever
    # normal retrieve (send queries)
    python retriever.py --host 127.0.0.1 --port 9090 --batch_size 32 --dim 512 --k 10 --nq 10
    # retrieve with lists (send queries & the list of centroids to search)
    python retriever.py --host 127.0.0.1 --port 9090 --batch_size 32 --dim 512 --k 10 --nq 10 --nprobe 32 --request_with_lists 1
"""
import torch
from typing import Optional, List
import socket
import select
import numpy as np

from ralm.retriever.serialization_utils import \
    answer_message_len, encode_request, encode_request_with_lists, decode_answer

class BaseRetriever:
    
    def __init__(self):
        raise NotImplementedError
    
    def retrieve(self):
        raise NotImplementedError
    
class DummyRetriever(BaseRetriever):

    def __init__(self, default_k=10):
        self.default_k = default_k
        

    def retrieve(self, query : np.array, k : Optional[int] = 10):
        """
        Input: 
            query -> torch.Tensor, shape: (batch_size, hidden_dim)

        Return some dummy output as a dict
            {"id" : [1, 10, 56, ...],  "dist": [.2, 1.2, 9.7, ...]}
            "id" : int64 -> flat array with shape batch_size * k
            "dist" : float32 -> flat array with shape batch_size * k
        """
        if k is None: 
            k = self.default_k

        # batch_size, hidden_dim = query.shape
        # query = query.cpu()

        # dummy_out = dict()
        # dummy_out["id"] = [0 for i in range(batch_size * k)]
        # dummy_out["dist"] = [0.0 for i in range(batch_size * k)]

        dummy_out = None
            
        return dummy_out

    def retrieve_with_lists(self, query : np.array, list_IDs : np.array):
        """
        Input:
            query -> numpy.Array, shape: (batch_size, hidden_dim)
            list_IDs -> numpy.Array, shape: (batch_size, nprobe)
        """
        assert query.shape[0] == list_IDs.shape[0]
        dummy_out = None
        return dummy_out

class ExternalRetriever(BaseRetriever):

    def __init__(self, host: Optional[str] = None, port: Optional[int] = 9090,
          batch_size : Optional[int] = 1, dim : Optional[int] = 512, default_k : Optional[int] = 10):
          
        self.default_k = default_k
        self.batch_size = batch_size
        self.dim = dim
        self.HOST = socket.gethostbyname(host) if host else socket.gethostbyname(socket.gethostname())
        self.PORT = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.poll_socket = select.poll()
        self.poll_socket.register(self.socket, select.POLLIN)

        self.socket.connect((self.HOST, self.PORT))
        print(f"Connected to {self.HOST}:{self.PORT}")

    def sync_with_coordinator(self):
        """
        Send a 4-byte message to the coordinator and wait for a 4-byte response
        
        This method can be used when multi-GPU needs to interact with the coordinator,
            where the coordinator tells each GPU when to start the inference
        """
        BYTE_ORDER_PY = 'big'
        dummy_one = int(1)

        print("Waiting for coordinator to start the inference...")
        msg_A = dummy_one.to_bytes(4, byteorder=BYTE_ORDER_PY)
        assert len(msg_A) == 4
        self.socket.send(msg_A)

        msg_B = self.socket.recv(4)
        signal = int.from_bytes(msg_B, byteorder=BYTE_ORDER_PY)
        assert signal == 1
        print("Received signal from coordinator, start inference now...")
                        
    def poll(self):
        """
        Poll the server to check if there is data ready to receive
        """
        poll_result = self.poll_socket.poll(0)
        return poll_result

    def retrieve_send(self, query : np.array, k : Optional[int] = None):
        """
        Send out a single query
        """
        if k is None: 
            k = self.default_k
        assert query.shape == (self.batch_size, self.dim)
        request = encode_request(query, k, self.batch_size, self.dim)
        # print(f"Sending request of size {len(request)} bytes")
        sent_bytes = 0
        while sent_bytes < len(request):
            sent_bytes += self.socket.send(request[sent_bytes:])
        
    def retrieve_with_lists_send(self, query : np.array, list_IDs : np.array, k : Optional[int] = None):
        """
        Send out a single query & the list of centroids to search

        Input:
            query -> numpy.Array, shape: (batch_size, hidden_dim)
            list_IDs -> numpy.Array, shape: (batch_size, nprobe)
        """
        if k is None: 
            k = self.default_k
            
        assert query.shape == (self.batch_size, self.dim)
        batch_size, nprobe = list_IDs.shape
        assert batch_size == self.batch_size

        request_with_lists = encode_request_with_lists(query, list_IDs, self.batch_size, self.dim, nprobe, k)
        # print(f"Sending request of size {len(request_with_lists)} bytes")
        sent_bytes = 0
        while sent_bytes < len(request_with_lists):
            sent_bytes += self.socket.send(request_with_lists[sent_bytes:])

    def retrieve_recv(self, k : Optional[int] = None):
        """
        Receive the answer to a single query
        """
        if k is None: 
            k = self.default_k
        results_len = answer_message_len(k, self.batch_size)
        # print("results_len:", results_len)
        answer = b''
        while len(answer) < results_len:
            answer += self.socket.recv(results_len - len(answer))
        # print(f"Received answer of size {len(answer)} bytes")
        indices, distances = decode_answer(answer, k, self.batch_size)
        return indices, distances
    
    def retrieve(self, query : np.array, k : Optional[int] = None):
        """
        Send out query and receive the answer
        """
        if k is None: 
            k = self.default_k
        self.retrieve_send(query, k)
        indices, distances = self.retrieve_recv(k)

        return indices, distances

    def retrieve_with_lists(self, query : np.array, list_IDs : np.array, k : Optional[int] = None):
        """
        Send out query & the list of centroids to search and receive the answer
        """
        if k is None:
            k = self.default_k
        self.retrieve_with_lists_send(query, list_IDs)
        indices, distances = self.retrieve_recv(k)
        
        return indices, distances
        
if __name__ == '__main__':
    
    import argparse 
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--nq', type=int, default=10, help="number of queries to send to the server")
    parser.add_argument('--nprobe', type=int, default=32, help="number of lists to search")
    parser.add_argument('--request_with_lists', type=int, default=0, help="whether to send the list of centroids to search")

    args = parser.parse_args()

    host = args.host
    port = args.port
    batch_size = args.batch_size
    dim = args.dim
    k = args.k
    nq = args.nq
    nprobe = args.nprobe
    request_with_lists = args.request_with_lists

    retriever = ExternalRetriever(default_k=k, host=host, port=port, batch_size=batch_size, dim=dim)

    # create and send some test data to the server
    query = np.random.rand(batch_size, dim).astype(np.float32)
    # query = torch.rand(batch_size, dim, dtype=torch.float32)

    list_IDs = np.random.randint(0, nprobe, size=(batch_size, nprobe), dtype=np.int32)
    list_centroids = np.random.rand(batch_size, nprobe, dim).astype(np.float32)

    for i in range(nq):
        if request_with_lists:
            indices, distances = retriever.retrieve_with_lists(query, list_IDs, k)
        else:
            indices, distances = retriever.retrieve(query, k)
        