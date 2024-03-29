"""

Example usage:

python print_server_coordinator_gpu_cmd.py --model_base_config config/Dec-S.yaml --coordinator_base_config config/coordinator.yaml --runtime_config config/ralm_runtime.yaml
"""

# model configuration yaml
import os
import yaml
import argparse 
import numpy as np

from utils import parse_arch

print("WARNING: Please tune runtime config in config/ralm_runtime.yaml before running this script.\n\n")

parser = argparse.ArgumentParser()

parser.add_argument('--model_base_config', type=str, default="config/Dec-S.yaml", help="path to yaml configuration file")
parser.add_argument('--coordinator_base_config', type=str, default="config/coordinator.yaml", help="path to yaml configuration file")
parser.add_argument('--runtime_config', type=str, default="config/ralm_runtime.yaml", help="path to yaml configuration file")
args = parser.parse_args()

config_dict = {}

with open(args.model_base_config, "r") as f:
    update_dict = yaml.safe_load(f)
    for key in update_dict:
        assert key not in config_dict, f"Duplicate key {key} in {args.model_base_config}"
    config_dict.update(update_dict)

with open(args.coordinator_base_config, "r") as f:
    update_dict = yaml.safe_load(f)
    for key in update_dict:
        assert key not in config_dict, f"Duplicate key {key} in {args.coordinator_base_config}"
    config_dict.update(update_dict)

with open(args.runtime_config, "r") as f:
    update_dict = yaml.safe_load(f)
    for key in update_dict:
        assert key not in config_dict, f"Duplicate key {key} in {args.runtime_config}"
    config_dict.update(update_dict)

def append(argname, var):
    if var is not None:
        return f' --{argname} {var} '
    else:
        return ' '

if config_dict["save_profiling"]:
	# latency profiling saving only supported with 1 GPU currently
	assert config_dict["llm_ngpus"] == 1

# CPU/GPU search
if "FPGA" not in config_dict["architecture"]: 

	parsed_device, parsed_omp_threads, parsed_ngpu = parse_arch(config_dict["architecture"])
	if parsed_device == 'CPU':
		assert config_dict['search_device'] == 'cpu'
		assert config_dict['search_omp_threads'] == parsed_omp_threads
	elif parsed_device == 'GPU':
		assert config_dict['search_device'] == 'gpu'
		assert config_dict['search_ngpu'] == parsed_ngpu

	search_cmd = 'python start_faiss_server.py' \
			+ append('model_base_config', args.model_base_config) \
			+ append('search_server_host', config_dict["search_server_host"]) \
			+ append('search_server_port', config_dict["search_server_port"]) \
			+ append('batch_size', config_dict["batch_size"]) \
			+ append('request_with_lists', config_dict["request_with_lists"]) \
			+ append('device', config_dict['search_device']) \
			+ append('ngpu', config_dict["search_ngpu"]) \
			+ append('omp_threads', config_dict["search_omp_threads"])

	print("Using CPU/GPU for vector search, search_cmd:\n", search_cmd)
# FPGA search
else:
	# python launch_CPU_and_FPGA.py --config_fname ./config/local_network_test_1_FPGA.yaml --mode CPU_coordinator --TOPK 100 --batch_size 1000 --total_batch_num 105
	assert config_dict["request_with_lists"] == 1

	if config_dict["model_name"] == 'Dec-S' or config_dict["model_name"] == 'EncDec-S':
		FPGA_search_config = 'RALM-S1000M_M32.yaml'
		assert config_dict["architecture"] == '1FPGA-1GPU'
	elif config_dict["model_name"] == 'Dec-L' or config_dict["model_name"] == 'EncDec-L':
		FPGA_search_config = 'RALM-L1000M_M64.yaml'
		assert config_dict["architecture"] == '2FPGA-1GPU'
	else:
		raise ValueError(f'Invalid model_name: {config_dict["model_name"]}')

	total_batch_num = int(np.ceil((config_dict["n_warmup_batch"] + config_dict["n_batch"]) * config_dict["seq_len"] / config_dict["retrieval_interval"]))
	if config_dict["use_tiktok"]:
		total_batch_num *= 2
	FPGA_coord_cmd = 'python launch_CPU_and_FPGA.py ' \
		f' --config_fname ./config/{FPGA_search_config}' \
		f' --mode CPU_coordinator --TOPK {config_dict["k"]} --batch_size {config_dict["batch_size"]} --total_batch_num {total_batch_num}'
	print("Using FPGA for vector search, the FPGA coordinator command:\n", FPGA_coord_cmd)

# GPU coordinator + GPU LLM
coord_gpu_cmd = f'python start_coordinator_and_GPU.py' \
                + append('model_base_config', args.model_base_config) \
                + append('coordinator_base_config', args.coordinator_base_config) \
                + append('search_server_host' , config_dict["search_server_host"]) \
                + append('search_server_port' , config_dict["search_server_port"]) \
                + append('retrieval_interval', config_dict["retrieval_interval"]) \
                + append('request_with_lists', config_dict["request_with_lists"]) \
                + append('batch_size', config_dict["batch_size"]) \
                + append('n_warmup_batch', config_dict["n_warmup_batch"]) \
                + append('n_batch', config_dict["n_batch"]) \
                + append('ngpu', config_dict["llm_ngpus"]) \
                + append('start_gpu_id', config_dict["llm_start_gpu_id"]) \
                + append('use_tiktok', config_dict["use_tiktok"]) \
                + append('save_profiling', config_dict["save_profiling"]) \
				+ append('architecture', config_dict["architecture"]) \
				+ append('profile_dir', config_dict["profile_dir"]) \
				+ append('profile_fname', config_dict["profile_fname"])	

print("\ncoord_gpu_cmd:\n", coord_gpu_cmd)
