"""
The RALM performance is stored in a hierachical dictionary: 
    d_perf = d[model_name][architecture][retrieval_interval][batch_size]
        (we assume [model_name][index][k][nprobe] are tied to a certain model)
        d_perf['latency_ms'] = [latency_array_1 (batch 1), latency_array_2 (batch 2), ...],
            np array, shape = (batch_size, seq_len)
        d_perf['throughput_tokens_per_sec'] = [throughput_1, throughput_2, ...],
            np array, shape = (batch_size,)
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

plt.style.use('ggplot')

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

latency_dict_baseline = load_obj('./performance_results_archive/', 'RALM_latency_throughput_CPU_GPU')
latency_dict_FPGA = load_obj('./performance_results_archive/', 'RALM_latency_throughput_FPGA')
batch_size = 1

models = ['Dec-S', 'EncDec-S']#, 'Dec-L', 'EncDec-L']

x_labels = []
ys_baseline = []
ys_FPGA = []

for model_name in models:

    if model_name == 'Dec-S': 
        CPU_architecture = '8CPU'
        FPGA_architecture = '1FPGA-1GPU'
        retrieval_intervals = [1]
    elif model_name == 'EncDec-S':
        CPU_architecture = '8CPU'
        FPGA_architecture = '1FPGA-1GPU'
        retrieval_intervals = [8, 64, 512]
    elif model_name == 'Dec-L':
        CPU_architecture = '16CPU'
        FPGA_architecture = '2FPGA-1GPU'
        retrieval_intervals = [1]
    elif  model_name == 'EncDec-L':
        CPU_architecture = '16CPU'
        FPGA_architecture = '2FPGA-1GPU'
        retrieval_intervals = [8, 64, 512]

    for retrieval_interval in retrieval_intervals:
        x_label = f'{model_name}\ninterval={retrieval_interval}' 
        y_baseline = np.sum(latency_dict_baseline[model_name][CPU_architecture][retrieval_interval][batch_size]['latency_ms'], axis=1) / 1000
        y_FPGA = np.sum(latency_dict_FPGA[model_name][FPGA_architecture][retrieval_interval][batch_size]['latency_ms'], axis=1) / 1000
        
        x_labels.append(x_label)
        ys_baseline.append(y_baseline)
        ys_FPGA.append(y_FPGA)

x = np.arange(len(x_labels))  # the label locations

ys_baseline = np.array(ys_baseline)
y_baseline_means = np.mean(ys_baseline, axis=1)
y_baseline_min = np.min(ys_baseline, axis=1)
y_baseline_max = np.max(ys_baseline, axis=1)

ys_FPGA = np.array(ys_FPGA)
y_FPGA_means = np.mean(ys_FPGA, axis=1)
y_FPGA_min = np.min(ys_FPGA, axis=1)
y_FPGA_max = np.max(ys_FPGA, axis=1)

yerr_baseline = [y_baseline_means - y_baseline_min, y_baseline_max - y_baseline_means]
yerr_FPGA = [y_FPGA_means - y_FPGA_min, y_FPGA_max - y_FPGA_means]

max_y_error = max(np.max(yerr_baseline), np.max(yerr_FPGA))
# TODO: make annotated text y-offset dependent on max_y_error

width = 0.35  # the width of the bars

# TODO: add error bars

fig, ax = plt.subplots()
rects1  = ax.bar(x - width/2, y_baseline_means, width)#, label='Men')
rects2 = ax.bar(x + width/2, y_FPGA_means, width)#, label='Women')

ax.errorbar(x - width/2, y_baseline_means, yerr=yerr_baseline, fmt='none', color='black', capsize=3)
ax.errorbar(x + width/2, y_FPGA_means, yerr=yerr_FPGA, fmt='none', color='black', capsize=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Per-sequence Latency (sec)')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend([rects1, rects2], ["CPU-GPU", "FPGA-GPU"], loc="lower center", ncol=1, \
  facecolor='white', framealpha=1, frameon=False, bbox_to_anchor=(0, 1, 1, 0.2))


def autolabel(rects, align='center'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        text = '{:.1f}'.format(height)

        if align == 'center':
            text_x = 0
        elif align == 'left':
            text_x = -len(text)
        elif align == 'right':
            text_x = len(text)
        ax.annotate(text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(text_x, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=0)


autolabel(rects1, align='center')
autolabel(rects2, align='center')

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./images/ralm_latency_per_sequence.png', transparent=False, dpi=200, bbox_inches="tight")
# plt.show()