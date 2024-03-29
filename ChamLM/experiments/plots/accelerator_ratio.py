"""
Load from a performance dictionary, and plot the performance

The RALM performance is stored in a hierachical dictionary: 
    d_perf = d[model_name][architecture][retrieval_interval][batch_size]
        (we assume [dbname][index][k][nprobe] are tied to a certain model)
        d_perf['latency_ms'] = [latency_array_1 (batch 1), latency_array_2 (batch 2), ...],
            np array, shape = (batch_size, seq_len)
        d_perf['throughput_tokens_per_sec'] = [throughput_1, throughput_2, ...],
            np array, shape = (batch_size,)

"""

# Given the Vector DB performance & the GPU performance, calculate the ratio between accelerators, 
#   i.e., how many GPUs are needed to match the performance of an FPGA-based vector DB

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# plt.style.use('ggplot')
plt.style.use('seaborn-pastel') 

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

ralm_GPU_latency_throughput_dict = load_obj('./performance_results_archive/', 'RALM_latency_throughput_CPU_GPU')
ralm_FPGA_latency_throughput_dict = load_obj('./performance_results_archive/', 'RALM_latency_throughput_FPGA')
vector_search_FPGA_throughput_dict = load_obj('./performance_results_archive/', 'vector_search_throughput_FPGA_dict')


x_labels_EncDec = []
y_baseline_means = []
y_FPGA_means = []

models = ['EncDec-S', ]#'EncDec-L']

def get_accelerator_ratio_EncDec(model_name, retrieval_intervals):

    # Get the GPU throughput, use the case of interval=512 as an approximate of no retrieval
    if model_name == 'EncDec-S':
        CPU_architecture = '8CPU'
        GPU_retrieval_interval = 512
        GPU_batch_size = 64

        dbname = 'RALM-S1000M'
        index_key = 'IVF32768,PQ32'
        FPGA_architecture = '1FPGA-1GPU'
        FPGA_batch_size = 64 # FPGA vec DB throughput is insensitive to batch size, thus use 64 as an approximate
        k = 100
        nprobe = 32

    elif  model_name == 'EncDec-L':
        CPU_architecture = '16CPU'
        GPU_retrieval_interval = 512
        GPU_batch_size = 8

        dbname = 'RALM-L1000M'
        index_key = 'IVF32768,PQ64'
        FPGA_architecture = '2FPGA-1GPU'
        FPGA_batch_size = 64 # FPGA vec DB throughput is insensitive to batch size, thus use 64 as an approximate
        k = 100
        nprobe = 32

    # tokens / sec = (batch_size * seq_len) / e2e_latency
    GPU_throughput = ralm_GPU_latency_throughput_dict[model_name][CPU_architecture][GPU_retrieval_interval][GPU_batch_size]['throughput_tokens_per_sec'].mean()
    # QPS = batch_size / e2e_latency
    FPGA_throughput = np.average(vector_search_FPGA_throughput_dict[dbname][index_key][FPGA_architecture][k][nprobe])
    # print(f'{model_name}, GPU_throughput={GPU_throughput}, FPGA_throughput={FPGA_throughput}')
      
    y = []
    for retrieval_interval in retrieval_intervals:
        # The needed GPU query throughput
        GPU_query_throughput = GPU_throughput / retrieval_interval
        num_GPU_to_saturate_FPGA = FPGA_throughput / GPU_query_throughput
        y.append(num_GPU_to_saturate_FPGA)
        print(f'{model_name}, interval={retrieval_interval}, num_GPU_to_saturate_FPGA={num_GPU_to_saturate_FPGA}')
    
    return y

def get_accelerator_ratio_Dec(model_name, retrieval_intervals):

    if model_name == 'Dec-S':
        CPU_architecture = '8CPU'
        GPU_retrieval_interval = 1
        GPU_batch_size = 64
        
        dbname = 'RALM-S1000M'
        index_key = 'IVF32768,PQ32'
        FPGA_architecture = '1FPGA-1GPU'
        FPGA_batch_size = 64 # FPGA vec DB throughput is insensitive to batch size, thus use 64 as an approximate
        k = 100
        nprobe = 32
        
    elif model_name == 'Dec-L':
        CPU_architecture = '16CPU'
        GPU_retrieval_interval = 1
        GPU_batch_size = 8
        
        dbname = 'RALM-L1000M'
        index_key = 'IVF32768,PQ64'
        FPGA_architecture = '2FPGA-1GPU'
        FPGA_batch_size = 64 # FPGA vec DB throughput is insensitive to batch size, thus use 64 as an approximate
        k = 100
        nprobe = 32


    # tokens / sec = (batch_size * seq_len) / e2e_latency
    e2e_throughput = ralm_FPGA_latency_throughput_dict[model_name][FPGA_architecture][GPU_retrieval_interval][GPU_batch_size]['throughput_tokens_per_sec'].mean()
    # QPS = batch_size / e2e_latency
    FPGA_throughput = np.average(vector_search_FPGA_throughput_dict[dbname][index_key][FPGA_architecture][k][nprobe])
    # GPU throughput can be derived by e2e_latency = GPU_latency + FPGA_latency -> 1 / e2e_throughput = 1 / GPU_throughput + 1 / FPGA_throughput
    GPU_throughput = 1 / (1 / e2e_throughput - 1 / FPGA_throughput)
    # print(f'{model_name}, e2e_throughput={e2e_throughput}, FPGA_throughput={FPGA_throughput}, GPU_throughput={GPU_throughput}')

    y = []
    for retrieval_interval in retrieval_intervals:
        # The needed GPU query throughput
        GPU_query_throughput = GPU_throughput / retrieval_interval
        num_GPU_to_saturate_FPGA = FPGA_throughput / GPU_query_throughput
        y.append(num_GPU_to_saturate_FPGA)
        print(f'{model_name}, interval={retrieval_interval}, num_GPU_to_saturate_FPGA={num_GPU_to_saturate_FPGA}')
    
    return y

if __name__ == '__main__':

    label_font = 10
    markersize = 8
    tick_font = 9
    legend_font = 10

    fig, ax = plt.subplots(figsize=(6, 2))

    retrieval_intervals_EncDec = [1, 2, 4, 8, 16, 32, 64]
    retrieval_intervals_Dec = [1]

    y_EncDec_S = get_accelerator_ratio_EncDec('EncDec-S', retrieval_intervals_EncDec)
    y_EncDec_L = get_accelerator_ratio_EncDec('EncDec-L', retrieval_intervals_EncDec)

    y_Dec_S = get_accelerator_ratio_Dec('Dec-S', retrieval_intervals_Dec)
    y_Dec_L = get_accelerator_ratio_Dec('Dec-L', retrieval_intervals_Dec)

    x_labels_EncDec = [str(interval) for interval in retrieval_intervals_EncDec]
    x_labels_Dec = [str(interval) for interval in retrieval_intervals_Dec]

    ax.plot(x_labels_Dec, y_Dec_S, linewidth=2, marker='v', markersize=markersize, label='Dec-S')
    ax.plot(x_labels_Dec, y_Dec_L, linewidth=2, marker='^', markersize=markersize, label='Dec-L')
    ax.plot(x_labels_EncDec, y_EncDec_S, linewidth=2, marker='+', markersize=markersize, label='EncDec-S')
    ax.plot(x_labels_EncDec, y_EncDec_L, linewidth=2, marker='x', markersize=markersize, label='EncDec-L')


    # ax.legend([plot_EncDec_S, plot_EncDec_L] , ["EncDec-S","EncDec-L"], loc='upper left')
    # plt.legend([plot_EncDec_S] , ["EncDec_S"], loc='upper left', fontsize=14)
    ax.legend(fontsize=legend_font, ncol=4, loc=(0.02, 1.02), frameon=False)

    ax.set_ylabel('Number of GPUs\nto saturate ChamVS', fontsize=label_font)
    ax.set_xlabel('Retrieval Interval', fontsize=label_font)
    # ax.set_ylim([0, 100])
    ax.set_yscale('log')

    plt.rcParams.update({'figure.autolayout': True})
    fig.tight_layout()
    plt.savefig(f'./images/ralm_accelerator_ratio.png', dpi=500)