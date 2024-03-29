"""
A server that receives a query and returns a random answer.

Example usage:

    python server.py --host 127.0.0.1 --port 9090 --batch_size 32 --dim 512 --k 10 --nprobe 32 --request_with_lists 0 --delay_ms 0
"""

import time
import torch
import socket

import numpy as np

from ralm.retriever.serialization_utils import request_message_length, request_message_length_with_lists, decode_request, decode_request_with_lists, encode_answer
from typing import Optional

class BaseServer:

    def __init__(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError


class RandomAnswerServer(BaseServer):

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, 
                 batch_size : Optional[int] = 1, dim : Optional[int] = 512, k : Optional[int] = 10,
                 nprobe : Optional[int] = 32, request_with_lists : Optional[int] = 0, delay_ms : Optional[int] = 0):
        """
        Args:
            host: str, ip address of the server
            port: int, port of the server
            batch_size: int, batch size of the queries
            dim: int, dimension of the queries
            delay_ms: int, query answer delay in milliseconds
            
        
            request_with_lists: bool, whether to receive queries with lists or not
                True: receive queries with lists (C2F format, IVF lists already scanned by Faiss)
                False: receive queries as is (queries with batch headers)
        """
        self.batch_size = batch_size
        self.dim = dim
        self.k = k
        self.nprobe = nprobe
        self.request_with_lists = request_with_lists
        self.delay_ms = delay_ms
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
                assert k == self.k
                assert list_IDs.shape == (self.batch_size, self.nprobe)
            else:
                k, queries = decode_request(encoded_queries, self.batch_size, self.dim)
                assert queries.shape == (self.batch_size, self.dim)
                assert k == self.k

            indices = np.array(list(range(self.batch_size * self.k)), dtype=np.int64)
            indices = indices.reshape(self.batch_size, self.k)

            distances = np.random.default_rng().standard_normal(size=(self.batch_size, self.k), dtype='float32')
            answer = encode_answer(indices, distances, self.k, self.batch_size)
            time.sleep(self.delay_ms / 1000)

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--request_with_lists", type=int, default=0)
    parser.add_argument("--delay_ms", type=int, default=0)

    args = parser.parse_args()
 
    host = args.host
    port = args.port
    batch_size = args.batch_size
    dim = args.dim
    k = args.k
    nprobe = args.nprobe
    request_with_lists = args.request_with_lists
    delay_ms = args.delay_ms
    
    random_answer_server = RandomAnswerServer(
        host=host, port=port, batch_size=batch_size, dim=dim, k=k,
        nprobe=nprobe, request_with_lists=request_with_lists, delay_ms=delay_ms)
    # random_answer_server.start_one_query_per_conn() # one query per connection
    random_answer_server.start() # one connection, multiple queries