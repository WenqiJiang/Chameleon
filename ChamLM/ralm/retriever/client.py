import torch
import socket

from ralm.retriever.serialization_utils import answer_message_len, encode_request, decode_answer

# Choose either hardcoded IP or socket.gethostname()
# HOST = socket.gethostbyname('127.0.1.1')
HOST = socket.gethostbyname(socket.gethostname())
PORT = 9090

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
socket.connect((HOST, PORT))

# create and send some test data to the server
batch_size = 32
dim = 128
print(f"batch size: {batch_size}, dim: {dim} : please make sure client and server have the same setting")
queries = torch.rand(batch_size, dim, dtype=torch.float32)
k = 2

socket.send(encode_request(queries, k, batch_size, dim))

indices, distances = decode_answer(socket.recv(answer_message_len(k, batch_size)), k, batch_size)

print(f"k: {k}")
print(f"queries: {queries}")
print(f"indices: {indices}")
print(f"distances: {distances}")