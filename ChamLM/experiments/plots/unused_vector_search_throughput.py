# Original plot: https://github.com/WenqiJiang/matplotlib-templates/blob/master/grouped_bar_with_error_bar/grouped_bar_with_error_bar.py

"""
Load from a performance dictionary, and plot the performance

Latency result format (in dict): [dbname][index][architecture][k][nprobe][batch_size] = latency_array (ms)
    dbname example: 'SIFT1000M'
    index example: 'IVF32768,PQ16'
    nprobe example: 32
    batch_size example: 64
    architecture example: '16CPU-1GPU' or '32CPU', here the number means the number of CPU cores and GPU cards
    latency_array example: np.array([1.5, 3.4, ...]) (ms)

    
Throughput result format (in dict): [dbname][index][architecture][k][nprobe] = throughput_array (QPS)
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# plt.style.use('ggplot')
plt.style.use('seaborn-colorblind') 

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def get_error_bar(d):
    """
    Given the key, return a dictionary of std deviation
    """
    dict_error_bar = dict()
    for key in d:
        array = d[key]
        dict_error_bar[key] = np.std(array)
    return dict_error_bar

def get_mean(d):
    """
    Given the key, return a dictionary of mean
    """
    dict_mean = dict()
    for key in d:
        array = d[key]
        dict_mean[key] = np.average(array)
    return dict_mean

def get_y_array(d, keys):
    """
    Given a dictionary, and a selection of keys, return an array of y value
    """
    y = []
    for key in keys:
        y.append(d[key])
    return y

throughput_dict = load_obj('./performance_results_archive/', 'vector_search_throughput_CPU_GPU_dict')
print(throughput_dict)

dbname_list = ['SIFT1000M', 'Deep1000M', 'RALM-S1000M', 'RALM-L1000M', 'RALM-S2000M', 'RALM-L2000M']
x_labels = dbname_list

index_key = 'IVF32768,PQ16'
k = 100
nprobe = 32

y_cpu_8 = {}
y_cpu_8_gpu = {}
y_cpu_16 = {}
y_cpu_16_gpu = {}
y_cpu_32 = {}

x_cpu_8 = []
x_cpu_8_gpu = []
x_cpu_16 = []
x_cpu_16_gpu = []
x_cpu_32 = []

width = 0.2

for i, dbname in enumerate(dbname_list):
    if dbname == 'SIFT1000M' or dbname == 'Deep1000M':
        index_key = 'IVF32768,PQ16'
        architectures = ['8CPU', '8CPU-1GPU', '16CPU']
    elif dbname == 'RALM-S1000M' or dbname == 'RALM-S2000M':
        index_key = 'IVF32768,PQ32'
        architectures = ['16CPU', '16CPU-1GPU', '32CPU']
    elif dbname == 'RALM-L1000M' or dbname == 'RALM-L2000M':
        index_key = 'IVF32768,PQ64'
        architectures = ['16CPU', '16CPU-1GPU', '32CPU']

    if '8CPU' in architectures:
        y_cpu_8[dbname] = throughput_dict[dbname][index_key]['8CPU'][k][nprobe]
    if '8CPU-1GPU' in architectures:
        y_cpu_8_gpu[dbname] = throughput_dict[dbname][index_key]['8CPU-1GPU'][k][nprobe]
    if '16CPU' in architectures:
        y_cpu_16[dbname] = throughput_dict[dbname][index_key]['16CPU'][k][nprobe]
    if '16CPU-1GPU' in architectures:
        y_cpu_16_gpu[dbname] = throughput_dict[dbname][index_key]['16CPU-1GPU'][k][nprobe]
    if '32CPU' in architectures:
        y_cpu_32[dbname] = throughput_dict[dbname][index_key]['32CPU'][k][nprobe]
    
    if architectures == ['8CPU', '8CPU-1GPU', '16CPU']:
        x_cpu_8.append(i - 3 * width)
        x_cpu_8_gpu.append(i - 2 * width)
        x_cpu_16.append(i - 1 * width)
    elif architectures == ['16CPU', '16CPU-1GPU', '32CPU']:
        x_cpu_16.append(i - 3 * width)
        x_cpu_16_gpu.append(i - 2 * width)
        x_cpu_32.append(i - 1 * width)


x_cpu_8 = np.array(x_cpu_8)
x_cpu_8_gpu = np.array(x_cpu_8_gpu)
x_cpu_16 = np.array(x_cpu_16)
x_cpu_16_gpu = np.array(x_cpu_16_gpu)
x_cpu_32 = np.array(x_cpu_32)

y_cpu_8_means = get_y_array(get_mean(y_cpu_8), y_cpu_8.keys())
y_cpu_8_gpu_means = get_y_array(get_mean(y_cpu_8_gpu), y_cpu_8_gpu.keys())
y_cpu_16_means = get_y_array(get_mean(y_cpu_16), y_cpu_16.keys())
y_cpu_16_gpu_means = get_y_array(get_mean(y_cpu_16_gpu), y_cpu_16_gpu.keys())
y_cpu_32_means = get_y_array(get_mean(y_cpu_32), y_cpu_32.keys())

y_cpu_8_error_bar = get_y_array(get_error_bar(y_cpu_8), y_cpu_8.keys())
y_cpu_8_gpu_error_bar = get_y_array(get_error_bar(y_cpu_8_gpu), y_cpu_8_gpu.keys())
y_cpu_16_error_bar = get_y_array(get_error_bar(y_cpu_16), y_cpu_16.keys())
y_cpu_16_gpu_error_bar = get_y_array(get_error_bar(y_cpu_16_gpu), y_cpu_16_gpu.keys())
y_cpu_32_error_bar = get_y_array(get_error_bar(y_cpu_32), y_cpu_32.keys())


fig, ax = plt.subplots()
print(x_cpu_8)
print(y_cpu_8_means)
rects_cpu_8  = ax.bar(x_cpu_8, y_cpu_8_means, width)#, label='Men')
rects_cpu_8_gpu = ax.bar(x_cpu_8_gpu , y_cpu_8_gpu_means, width)#, label='Women')
rects_cpu_16 = ax.bar(x_cpu_16, y_cpu_16_means, width)
rects_cpu_16_gpu = ax.bar(x_cpu_16_gpu, y_cpu_16_gpu_means, width)
rects_cpu_32 = ax.bar(x_cpu_32, y_cpu_32_means, width)

ax.errorbar(x_cpu_8, y_cpu_8_means, yerr=y_cpu_8_error_bar, fmt=',', ecolor='black', capsize=1.5)
ax.errorbar(x_cpu_8_gpu, y_cpu_8_gpu_means, yerr=y_cpu_8_gpu_error_bar, fmt=',', ecolor='black', capsize=1.5)
ax.errorbar(x_cpu_16, y_cpu_16_means, yerr=y_cpu_16_error_bar, fmt=',', ecolor='black', capsize=1.5)
ax.errorbar(x_cpu_16_gpu, y_cpu_16_gpu_means, yerr=y_cpu_16_gpu_error_bar, fmt=',', ecolor='black', capsize=1.5)
ax.errorbar(x_cpu_32, y_cpu_32_means, yerr=y_cpu_32_error_bar, fmt=',', ecolor='black', capsize=1.5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Throughput (QPS)')
# ax.set_title('Scores by group and gender')
x = np.arange(len(x_labels))  # the label locations
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend([rects_cpu_8, rects_cpu_8_gpu, rects_cpu_16, rects_cpu_16_gpu, rects_cpu_32], ['8CPU', '8CPU-1GPU', '16CPU', '16CPU-1GPU', '32CPU'], loc="upper right", ncol=2, \
  facecolor='white', framealpha=1, frameon=False)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)


autolabel(rects_cpu_8)
autolabel(rects_cpu_8_gpu)
autolabel(rects_cpu_16)
autolabel(rects_cpu_16_gpu)
autolabel(rects_cpu_32)

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./images/vector_search_throughput.png', transparent=False, dpi=200, bbox_inches="tight")
# plt.show()