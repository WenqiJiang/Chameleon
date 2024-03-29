"""
This process starts a Faiss server, which loads some base configuration from the model configuration yaml.

Example usage:
	python start_faiss_server.py --model_base_config config/Dec-S.yaml \
		--search_server_host "127.0.0.1" --search_server_port "9091" \
		--batch_size 32 --request_with_lists 0 --device 'cpu' --ngpu 1 --omp_threads 8

"""


import argparse 
import yaml

from ralm.server.faiss_server import FaissServer

parser = argparse.ArgumentParser()
# model configuration yaml
parser.add_argument('--model_base_config', type=str, default="config/Dec-S.yaml", help="address of the base configuration")

parser.add_argument("--search_server_host", type=str, default="127.0.0.1")
parser.add_argument("--search_server_port", type=int, default=9091)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--request_with_lists", type=int, default=0)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--omp_threads", type=int, default=None)


args = parser.parse_args()

search_server_host = args.search_server_host
search_server_port = args.search_server_port
batch_size = args.batch_size
request_with_lists = args.request_with_lists
device = args.device
ngpu = args.ngpu
omp_threads = args.omp_threads


# yaml inputs
k = None
dim = None
nprobe = None

dbname = None
index_key = None
index_dir = None

# update local variables
config_dict = {}
with open(args.model_base_config, "r") as f:
    config_dict.update(yaml.safe_load(f))
locals().update(config_dict)

faiss_server = FaissServer(
	host=search_server_host, port=search_server_port,
	index_dir=index_dir, dbname=dbname, index_key=index_key, default_k=k, nprobe=nprobe,
	batch_size=batch_size, dim=dim, request_with_lists=request_with_lists, device=device, ngpu=ngpu, omp_threads=omp_threads)
faiss_server.start()