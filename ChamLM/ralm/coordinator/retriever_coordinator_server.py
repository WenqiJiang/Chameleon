"""
Given multiple GPUs, each with a single model & single GPU & a single external retriever, 
    this process consumes the requests from the GPU processes, and forward them to the CPU index server,
    the top-K results received from the CPU index server are also multiplexed to the original GPU processes.

Example usage:
    # one FPGA coordinator, two GPU processes
    python retriever_coordinator_server.py --search_server_host 127.0.0.1 --search_server_port 9091 \
        --local_port 9090 --ngpus 2 --batch_size 32 --num_queries_per_gpu 1024 --k 10 --dim 512 --nprobe 32 --request_with_lists 0

    # two FPGA coordinators, two GPU processes
    python retriever_coordinator_server.py --search_server_host '127.0.0.1 127.0.0.1' --search_server_port '9091 9092' \
        --local_port 9090 --ngpus 2 --batch_size 32 --num_queries_per_gpu 1024 --k 10 --dim 512 --nprobe 32 --request_with_lists 0
"""

import torch
import time
import numpy as np
import socket
import select

from ralm.retriever.serialization_utils import request_message_length, request_message_length_with_lists, answer_message_len, decode_request, encode_answer
from typing import Optional


class RetrieveCoordinator:

    def __init__(self, 
                 search_server_host = ['127.0.0.1'], search_server_port = [9091],
                 local_host: Optional[str] = None, local_port: Optional[int] = None, 
                 ngpus: Optional[int] = 1, batch_size : Optional[int] = 1, num_queries_per_gpu : Optional[int] = 1,
                 k :  Optional[int] = 10, dim : Optional[int] = 512, nprobe : Optional[int] = 32, request_with_lists : Optional[int] = 0):
        """
        Server start order:
            1. Start the CPU index server (skip when using start_dummy_answer())
            2. Start this process, make sure to correctly choose between start_dummy_answer() and start()
            3. Start the GPU processes, one by one
  
        request_with_lists: bool, whether to receive queries with lists or not
            True: receive queries with lists (C2F format, IVF lists already scanned by Faiss)
            False: receive queries as is (queries with batch headers)      
        """
        
        self.ngpus = ngpus
        self.batch_size = batch_size
        self.num_queries_per_gpu = num_queries_per_gpu
        self.k = k
        self.dim = dim
        self.nprobe = nprobe
        self.request_with_lists = request_with_lists
        
        assert len(search_server_host) == len(search_server_port)
        self.n_FPGA_coord = len(search_server_host)

        self.LOCAL_PORT = local_port if local_port else 9090
        self.LOCAL_HOST = socket.gethostbyname(
            local_host) if local_host else socket.gethostbyname(socket.gethostname())
        
        self.coordinator_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.coordinator_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.coordinator_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.coordinator_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.coordinator_sock.bind((self.LOCAL_HOST, self.LOCAL_PORT))

        self.search_server_HOST = [socket.gethostbyname(search_server_host[i]) for i in range(self.n_FPGA_coord)] 
        self.search_server_PORT = search_server_port 

        self.search_server_sock = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(self.n_FPGA_coord)]
        for i in range(self.n_FPGA_coord):
            self.search_server_sock[i].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.search_server_sock[i].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.search_server_sock[i].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        print("Server start listening...", flush=True)
        self.coordinator_sock.listen(self.ngpus)  # limit to n parallel connections
              
        if self.request_with_lists:
            self.query_msg_len = request_message_length_with_lists(self.batch_size, self.dim, self.nprobe)
        else:
            self.query_msg_len = request_message_length(self.batch_size, self.dim)
        print(f"Query message length is {self.query_msg_len} bytes", flush=True)

        self.result_msg_len = answer_message_len(self.k, self.batch_size)
        print(f"Result message length is {self.result_msg_len} bytes", flush=True)

    def accept_connections(self):
        """
        Accepts connections from the GPU processes.
        """
        self.communication_sockets = []
        self.addresses = []
        self.poll_gpu_objects = []
        self.per_gpu_recv_query_cnt = [0 for _ in range(self.ngpus)]
        self.query_gpu_ids = [[-1 for _ in range(self.ngpus * self.num_queries_per_gpu)] for _ in range(self.n_FPGA_coord)]

        for i in range(self.ngpus):
            communication_socket, address = self.coordinator_sock.accept()
            self.communication_sockets.append(communication_socket)
            self.addresses.append(address)
            poll = select.poll()
            poll.register(communication_socket, select.POLLIN)
            self.poll_gpu_objects.append(poll)
            print("Connection from", address, flush=True)

        print("Start synchronizing with the GPUs...", flush=True)          
        BYTE_ORDER_PY = 'big'
        dummy_one = int(1)

        # receive sync signal from the GPU processes
        for i in range(self.ngpus):
            msg_A = self.communication_sockets[i].recv(4)
            signal = int.from_bytes(msg_A, byteorder=BYTE_ORDER_PY)
            assert signal == 1
            print(f"Received sync signal from GPU {i}", flush=True)
            
        # send out sync signal to the GPU processes
        for i in range(self.ngpus):
            msg_B = dummy_one.to_bytes(4, byteorder=BYTE_ORDER_PY)
            assert len(msg_B) == 4
            self.communication_sockets[i].send(msg_B)
                        
        print("Sent out start signal, start coordination now...", flush=True)
                        

    def connect_to_search_server(self):
        """
        Connects to the CPU index server.
        """
        self.poll_cpu_search_server = [select.poll() for _ in range(self.n_FPGA_coord)]
        for i in range(self.n_FPGA_coord):
            self.search_server_sock[i].connect((self.search_server_HOST[i], self.search_server_PORT[i]))
            self.poll_cpu_search_server[i].register(self.search_server_sock[i], select.POLLIN)
            print(f"Connected to {self.search_server_HOST[i]}:{self.search_server_PORT[i]}", flush=True)
        
        self.per_gpu_recv_result_cnt = [0 for _ in range(self.ngpus)]
        self.assign_coord_cnt = 0

    def start_dummy_answer(self):
        """
        This is a dummy answer server, which will return random results to the GPU processes.
        """

        self.accept_connections()

        """
        Must track the number of queries processed per GPU,
         because poll will return >0 if there is a read event or IF THE SOCKET IS DISCONNECTED,
         afterwhile a call on recv will trigger an exception.
         https://stackoverflow.com/questions/17692447/does-poll-system-call-know-if-remote-socket-closed-or-disconnected
        """

        while True:
            
            for i in range(self.ngpus):
                
                # receive queries from the GPU processes
                if self.per_gpu_recv_query_cnt[i] == self.num_queries_per_gpu:
                    continue
                else:
                    poll_result = self.poll_gpu_objects[i].poll(0)
                    if poll_result:
                        # print(f"Poll from connection {i}")
                        encoded_queries = b''
                        while len(encoded_queries) < self.query_msg_len:
                            encoded_queries += self.communication_sockets[i].recv(self.query_msg_len - len(encoded_queries))
                        self.per_gpu_recv_query_cnt[i] += 1

                        if self.request_with_lists:
                            # no decoding func here, FPGA will decode it
                            pass
                        else:
                            k, queries = decode_request(encoded_queries, self.batch_size, self.dim)
                            assert queries.shape == (self.batch_size, self.dim)
                            assert k == self.k

                        #print(f"Message from client is: {message}")

                        indices = np.array(list(range(self.batch_size * self.k)), dtype=np.int64)
                        indices = indices.reshape(self.batch_size, self.k)
                        distances = np.random.default_rng().standard_normal(size=(self.batch_size, self.k), dtype='float32')
                        answer = encode_answer(indices, distances, self.k, self.batch_size)
                        # print(f"Length of answer is {len(answer)} bytes")
                        sent_bytes = 0
                        while sent_bytes < self.result_msg_len:
                            sent_bytes += self.communication_sockets[i].send(answer[sent_bytes:])


            all_finish = True
            for i in range(self.ngpus):
                if self.per_gpu_recv_query_cnt[i] < self.num_queries_per_gpu:
                    all_finish = False
                    break
            if all_finish:
                break
                # communication_socket.close()
                # #print(f"Connection with {address} ended!")

    def start(self):
        """
        forward the the GPU queries to the CPU index server, and multiplex the results back to the GPU processes.
        """

        received_query_cnt = 0
        received_query_cnt_per_FPGA_coord = [0 for _ in range(self.n_FPGA_coord)]
        received_result_cnt_per_FPGA_coord = [0 for _ in range(self.n_FPGA_coord)]
        
        self.connect_to_search_server()
        self.accept_connections()

        """
        Must track the number of queries processed per GPU,
         because poll will return >0 if there is a read event or IF THE SOCKET IS DISCONNECTED,
         afterwhile a call on recv will trigger an exception.
         https://stackoverflow.com/questions/17692447/does-poll-system-call-know-if-remote-socket-closed-or-disconnected
        """

        t_start = time.time()
        while True:
            
            for i in range(self.ngpus):
                
                # receive queries from the GPU processes
                # forward data from one GPU as long as this connection has data to send
                while True:
                    if self.per_gpu_recv_query_cnt[i] == self.num_queries_per_gpu:
                        break
                    poll_result = self.poll_gpu_objects[i].poll(0)
                    if poll_result:
                        # print("poll results: ", poll_result)
                        print("recv from GPU ", i)
                        encoded_queries = b''

                        while len(encoded_queries) < self.query_msg_len:
                            encoded_queries += self.communication_sockets[i].recv(self.query_msg_len - len(encoded_queries))
                        
                        assign_coord_id = received_query_cnt % self.n_FPGA_coord
                        print("send to FPGA coord ", assign_coord_id)

                        self.per_gpu_recv_query_cnt[i] += 1
                        self.query_gpu_ids[assign_coord_id][received_query_cnt_per_FPGA_coord[assign_coord_id]] = i
                        received_query_cnt += 1
                        received_query_cnt_per_FPGA_coord[assign_coord_id] += 1

                        sent_bytes = 0
                        while sent_bytes < self.query_msg_len:
                            sent_bytes += self.search_server_sock[assign_coord_id].send(encoded_queries[sent_bytes:])
                        # print(f"Finish forwarded query count {received_query_cnt}")
                    else:
                        break
        
            # receive results from the CPU index server
            while True:
                # pool until no results from the CPU index server
                all_result_poll_empty = True
                for i in range(self.n_FPGA_coord): 
                        
                    poll_result = self.poll_cpu_search_server[i].poll(0)
                    if poll_result:
                        print("Receive result from FPGA coord ", i)
                        all_result_poll_empty = False
                        # print("Poll from index server")
                        encoded_results = b''
                        while len(encoded_results) < self.result_msg_len:
                            encoded_results += self.search_server_sock[i].recv(self.query_msg_len - len(encoded_results))
                        gpu_id = self.query_gpu_ids[i][received_result_cnt_per_FPGA_coord[i]]
                        # print("Sending results to GPU", gpu_id)
                        sent_bytes = 0
                        while sent_bytes < self.result_msg_len:
                            sent_bytes += self.communication_sockets[gpu_id].send(encoded_results[sent_bytes:])
                        self.per_gpu_recv_result_cnt[gpu_id] += 1
                        received_result_cnt_per_FPGA_coord[i] += 1
                        # print("Finish result count", self.per_gpu_recv_result_cnt[gpu_id], received_result_cnt)
                if all_result_poll_empty:
                    break

            # TODO: edit
            if self.num_queries_per_gpu * self.ngpus == received_query_cnt and \
                self.num_queries_per_gpu * self.ngpus == np.sum(received_result_cnt_per_FPGA_coord):
                break

        t_end = time.time()
        print("Total time: {:.2f} ms".format((t_end - t_start) * 1000))
        print("Total query batches: {}\t batch_size: {}".format(received_query_cnt, self.batch_size))
        print("batches / sec:", received_query_cnt / (t_end - t_start))
        print("Queries / sec:", received_query_cnt * self.batch_size / (t_end - t_start))

if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser()

    parser.add_argument('--search_server_host', type=str, default="127.0.0.1 127.0.0.1", help="space separated list of index server hosts")
    parser.add_argument('--search_server_port', type=str, default="9091 9092", help="space separated list of index server ports")
    parser.add_argument('--local_host', type=str, default="127.0.0.1")
    parser.add_argument('--local_port', type=int, default=9090)
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_queries_per_gpu', type=int, default=10)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--request_with_lists", type=int, default=0)

    args = parser.parse_args()

    search_server_host = args.search_server_host
    search_server_host = search_server_host.split(" ")
    search_server_port = args.search_server_port
    search_server_port = search_server_port.split(" ")
    search_server_port = [int(port) for port in search_server_port]
    assert len(search_server_host) == len(search_server_port)
    local_host = args.local_host
    local_port = args.local_port
        
    ngpus = args.ngpus
    batch_size = args.batch_size
    num_queries_per_gpu = args.num_queries_per_gpu
    k = args.k
    dim = args.dim
    nprobe = args.nprobe
    request_with_lists = args.request_with_lists
        
    coordinator = RetrieveCoordinator(
        search_server_host=search_server_host, search_server_port=search_server_port,
        local_host=local_host, local_port=local_port,
        ngpus=ngpus, batch_size=batch_size, num_queries_per_gpu=num_queries_per_gpu, 
        k=k, dim=dim, nprobe=nprobe, request_with_lists=request_with_lists)
    
    # coordinator.start_dummy_answer() # return random results
    coordinator.start() # return from the CPU index server