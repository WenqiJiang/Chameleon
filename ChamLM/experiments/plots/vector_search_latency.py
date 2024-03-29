"""
Latency result format (in dict): [dbname][index][architecture][k][nprobe][batch_size] = latency_array (ms)
    dbname example: 'SIFT1000M'
    index example: 'IVF32768,PQ16'
    nprobe example: 32
    batch_size example: 64
    architecture example: '16CPU-1GPU' or '32CPU', here the number means the number of CPU cores and GPU cards
    latency_array example: np.array([1.5, 3.4, ...]) (ms)

    
Throughput result format (in dict): [dbname][index][architecture][k][nprobe] = throughput_array (QPS)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns 

import os
import pickle

# Seaborn version >= 0.11.0

# sns.set_theme(style="whitegrid")
# Set the palette to the "pastel" default palette:
# sns.set_palette("pastel")

# plt.style.use('seaborn-pastel') 

def get_default_colors():

  default_colors = []
  for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
      default_colors.append(color["color"])
      print(color["color"], type(color["color"]))

  return default_colors

default_colors = get_default_colors()
print("colors length:", len(default_colors))
# print(default_colors[0], type(default_colors[0]))

### Note: For violin graph, a single violin's data must be in the same column
###   e.g., given 3 violin plots, each with 100 points, the shape of the array
###   should be (100, 3), where the first column is for the first violin and so forth
# fake up some data


# Wenqi: flatten the table to a table. It's a dictionary with the key as schema.
#   The value of each key is an array.
# label category data
# xxx   xxx      xxx
# yyy   yyy      yyy

k = 100
nprobe = 32
batch_sizes = [1, 4, 16, 64]
dbname_list = ['SIFT1000M', 'Deep1000M', 'RALM-S1000M', 'RALM-L1000M']

# python color palette: https://matplotlib.org/stable/gallery/color/named_colors.html

colors = {
          # Deep, SIFT, SYN-512
          '8CPU': default_colors[0],
          '8CPU-1GPU': default_colors[1],
          '1FPGA-8CPU': default_colors[2],
          '1FPGA-1GPU': default_colors[3],

          # SYN-1024
          '16CPU': default_colors[4],
          '16CPU-1GPU': default_colors[5],
          '2FPGA-8CPU': default_colors[6],
          '2FPGA-1GPU': default_colors[7],

          # unused
          '32CPU': default_colors[8],

          'box_plot': 'k',
          'median_bubble': 'white',
          }

def get_architectures_and_index_keys(dbname):
    if dbname == 'SIFT1000M' or dbname == 'Deep1000M':
        index_key = 'IVF32768,PQ16'
        architectures = ['8CPU', '8CPU-1GPU', '1FPGA-8CPU', '1FPGA-1GPU']
    elif dbname == 'RALM-S1000M':
        index_key = 'IVF32768,PQ32'
        architectures = ['8CPU', '8CPU-1GPU', '1FPGA-8CPU', '1FPGA-1GPU']
    # elif dbname == 'RALM-S2000M':
    #     index_key = 'IVF32768,PQ32'
    #     architectures = ['16CPU', '16CPU-1GPU', '2FPGA-8CPU', '2FPGA-1GPU']
    elif dbname == 'RALM-L1000M':
        index_key = 'IVF32768,PQ64'
        architectures = ['16CPU', '16CPU-1GPU', '2FPGA-8CPU', '2FPGA-1GPU']
    # elif dbname == 'RALM-L2000M':
    #     index_key = 'IVF32768,PQ64'
    #     architectures = ['16CPU', '16CPU-1GPU', '4FPGA-8CPU', '4FPGA-1GPU']
    return architectures, index_key


def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def plot_seaborn(dbname):

    architectures, index_key = get_architectures_and_index_keys(dbname)

    d = {}
    d['label'] = [] # batch size
    d['data'] = []
    d['category'] = [] # architecture

    latency_dict_baseline = load_obj('./performance_results_archive/', 'vector_search_latency_CPU_GPU_dict')
    latency_dict_FPGA = load_obj('./performance_results_archive/', 'vector_search_latency_FPGA_dict')

    for arch in architectures:
        for batch_size in batch_sizes:
            if 'FPGA' in arch:
                latency_array = latency_dict_FPGA[dbname][index_key][arch][k][nprobe][batch_size]
            else:
                latency_array = latency_dict_baseline[dbname][index_key][arch][k][nprobe][batch_size]
            for latency in latency_array:
                d['label'].append(batch_size)
                d['data'].append(latency)
                d['category'].append(arch)
            
    df = pd.DataFrame(data=d)
    print(df.index)
    print(df.columns)

    plt.figure(figsize=(6, 1.5))
    # API: https://seaborn.pydata.org/generated/seaborn.violinplot.html
    # inner{“box”, “quartile”, “point”, “stick”, None}, optional
    # scale : {“area”, “count”, “width”}, optional
    ax = sns.violinplot(data=df, scale='width', inner="box", x="label", y="data", hue="category")

    legend_font = 10
    tick_font = 10
    label_font = 12

    x_tick_labels = [str(batch_size) for batch_size in batch_sizes]
    ax.set_xticklabels(x_tick_labels, fontsize=tick_font)
    # ax.set_yticklabels(y_tick_labels)
    plt.yticks(fontsize=tick_font)
    # # ax.ticklabel_format(axis='both', style='sci')
    ax.set_yscale("log")
    loc = (0.02,1.02)#"upper left" # "best"
    ax.legend(loc=loc, ncol=3, fontsize=legend_font, frameon=False)

    ax.tick_params(length=0, top=False, bottom=False, left=False, right=False, 
        labelleft=True, labelbottom=True, labelsize=12)
    ax.set_xlabel('Batch size', fontsize=label_font)#, labelpad=10)
    ax.set_ylabel('Latency (ms)', fontsize=label_font)#, labelpad=10)

    # ax.set_title(f'{dbname}', fontsize=label_font, y=1.35)
    ax.text(2.5, 5, f'{dbname}', fontsize=label_font)
    # plt.text(2, len(y_tick_labels) + 2, "Linear Heatmap", fontsize=16)

    plt.savefig(f'./images/latency_{dbname}.png', transparent=False, dpi=200, bbox_inches="tight")

    # plt.show()

min_speedup_median_cpu_gpu_to_cpu = 99999999
min_speedup_median_cpu_fpga_to_cpu = 99999999
min_speedup_median_gpu_fpga_to_cpu = 99999999
min_speedup_median_gpu_fpga_to_cpu_fpga = 99999999
min_speedup_p95_cpu_gpu_to_cpu = 99999999
min_speedup_p95_cpu_fpga_to_cpu = 99999999
min_speedup_p95_gpu_fpga_to_cpu = 99999999
min_speedup_p95_gpu_fpga_to_cpu_fpga = 99999999

max_speedup_median_cpu_gpu_to_cpu = 0
max_speedup_median_cpu_fpga_to_cpu = 0
max_speedup_median_gpu_fpga_to_cpu = 0
max_speedup_median_gpu_fpga_to_cpu_fpga = 0
max_speedup_p95_cpu_gpu_to_cpu = 0
max_speedup_p95_cpu_fpga_to_cpu = 0
max_speedup_p95_gpu_fpga_to_cpu = 0
max_speedup_p95_gpu_fpga_to_cpu_fpga = 0

def plot_matplotlib(dbname):
    
    label_font = 10
    markersize = 8
    tick_font = 8
    legend_font = 9

    # https://matplotlib.org/stable/gallery/statistics/customized_violin.html
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value


    def set_axis_style(ax, inds, label_positions, labels):
        ax.set_xticks(label_positions, labels=labels)
        ax.tick_params(which='major', width=0, length=0)
        ax.set_xlim(0.25, inds[-1] + 0.75)
        ax.set_xlabel('Batch size', fontsize=label_font)#, labelpad=10)
        ax.set_ylabel('Latency (ms)', fontsize=label_font)#, labelpad=10)
        ax.set_xticklabels(labels, fontsize=tick_font)
        plt.yticks(fontsize=tick_font)

    architectures, index_key = get_architectures_and_index_keys(dbname)

    latency_dict_baseline = load_obj('./performance_results_archive/', 'vector_search_latency_CPU_GPU_dict')
    latency_dict_FPGA = load_obj('./performance_results_archive/', 'vector_search_latency_FPGA_dict')

    architecture_labels = []
    batch_size_labels = []

    alpha_violin_plot = 0.8 
    relative_distance_between_plot_groups = 1.25

    data = []
    inds = [] # x positions of the violin charts
    curr_x_position = 1

    labels = []
    label_positions = []
    label_position = 0

    for batch_size in batch_sizes:
        for architecture in architectures:
            if 'FPGA' in architecture:
                latency_array = latency_dict_FPGA[dbname][index_key][architecture][k][nprobe][batch_size]
            else:
                latency_array = latency_dict_baseline[dbname][index_key][architecture][k][nprobe][batch_size]
            architecture_labels.append(architecture)
            batch_size_labels.append(batch_sizes)
            data.append(np.sort(latency_array.flatten()))
            inds.append(curr_x_position)
            label_position += curr_x_position
            curr_x_position += 1
        curr_x_position += relative_distance_between_plot_groups - 1
        label_position /= len(architectures)
        labels.append(f"{batch_size}")
        label_positions.append(label_position)
        label_position = 0
    
    inds = np.array(inds)


    fig, ax = plt.subplots(figsize=(6, 2))
    ax.set_yscale('log')

    parts = ax.violinplot(data, inds, vert=True, showmeans=False, showmedians=False, showextrema=False, widths=3/len(architectures)) 

    for pc, architecture_label in zip(parts['bodies'], architecture_labels):
        pc.set_facecolor(colors[architecture_label])
        pc.set_edgecolor(colors[architecture_label])
        # pc.set_edgecolor('black')
        pc.set_linewidth(0.6)
        pc.set_alpha(alpha_violin_plot)

    # shape of lists like median or p95: n_architectures (4) * n_batch_sizes (4), e.g.,
    # [16.20416641 12.93911934  1.92980766  1.43843889 17.1456337  16.10275507
    # 3.44537497  3.29347849 27.80307531 26.691854   10.82092524  9.40747261
    # 89.68430758 87.02380657 44.3975091  34.83526707]
    quartile1, medians, quartile3, p95 = np.percentile(data, [25, 50, 75, 95], axis=1)
    medians = np.median(data, axis=1)
    assert not np.allclose(medians, p95)

    print("==== db:", dbname, " ====")
    assert len(architectures) == 4   

    for i, batch_size in enumerate(batch_sizes):

        median_cpu = medians[i*len(architectures)]
        median_cpu_gpu = medians[i*len(architectures)+1]
        median_cpu_fpga = medians[i*len(architectures)+2]
        median_gpu_fpga = medians[i*len(architectures)+3]

        p95_cpu = p95[i*len(architectures)]
        p95_cpu_gpu = p95[i*len(architectures)+1]
        p95_cpu_fpga = p95[i*len(architectures)+2]
        p95_gpu_fpga = p95[i*len(architectures)+3]

        print("Batch size:", batch_size)
        print("\tMedian latency (ms): CPU: {:.2f}\tCPU-GPU: {:.2f}\tCPU-FPGA: {:.2f}\tGPU-FPGA: {:.2f}".format(
            median_cpu, median_cpu_gpu, median_cpu_fpga, median_gpu_fpga))
        print("\t95th percentile latency (ms): CPU: {:.2f}\tCPU-GPU: {:.2f}\tCPU-FPGA: {:.2f}\tGPU-FPGA: {:.2f}".format(
            p95_cpu, p95_cpu_gpu, p95_cpu_fpga, p95_gpu_fpga))
        speedup_median_cpu_gpu_to_cpu = median_cpu / median_cpu_gpu
        speedup_median_cpu_fpga_to_cpu = median_cpu / median_cpu_fpga
        speedup_median_gpu_fpga_to_cpu = median_cpu / median_gpu_fpga
        speedup_median_gpu_fpga_to_cpu_fpga = median_cpu_fpga / median_gpu_fpga
        speedup_p95_cpu_gpu_to_cpu = p95_cpu / p95_cpu_gpu
        speedup_p95_cpu_fpga_to_cpu = p95_cpu / p95_cpu_fpga
        speedup_p95_gpu_fpga_to_cpu = p95_cpu / p95_gpu_fpga
        speedup_p95_gpu_fpga_to_cpu_fpga = p95_cpu_fpga / p95_gpu_fpga
        print("\tMedian speedup: CPU-GPU/CPU: {:.2f}\tCPU-FPGA/CPU: {:.2f}\tGPU-FPGA/CPU: {:.2f}\tGPU-FPGA/CPU-FPGA: {:.2f}".format(
            speedup_median_cpu_gpu_to_cpu, speedup_median_cpu_fpga_to_cpu, speedup_median_gpu_fpga_to_cpu, speedup_median_gpu_fpga_to_cpu_fpga))
        print("\t95th percentile speedup: CPU-GPU/CPU: {:.2f}\tCPU-FPGA/CPU: {:.2f}\tGPU-FPGA/CPU: {:.2f}\tGPU-FPGA/CPU-FPGA: {:.2f}".format(
            speedup_p95_cpu_gpu_to_cpu, speedup_p95_cpu_fpga_to_cpu, speedup_p95_gpu_fpga_to_cpu, speedup_p95_gpu_fpga_to_cpu_fpga))
        
        global min_speedup_median_cpu_gpu_to_cpu, min_speedup_median_cpu_fpga_to_cpu, min_speedup_median_gpu_fpga_to_cpu, min_speedup_median_gpu_fpga_to_cpu_fpga
        global max_speedup_median_cpu_gpu_to_cpu, max_speedup_median_cpu_fpga_to_cpu, max_speedup_median_gpu_fpga_to_cpu, max_speedup_median_gpu_fpga_to_cpu_fpga
        min_speedup_median_cpu_gpu_to_cpu = np.amin([min_speedup_median_cpu_gpu_to_cpu, speedup_median_cpu_gpu_to_cpu])
        min_speedup_median_cpu_fpga_to_cpu = np.amin([min_speedup_median_cpu_fpga_to_cpu, speedup_median_cpu_fpga_to_cpu])
        min_speedup_median_gpu_fpga_to_cpu = np.amin([min_speedup_median_gpu_fpga_to_cpu, speedup_median_gpu_fpga_to_cpu])
        min_speedup_median_gpu_fpga_to_cpu_fpga = np.amin([min_speedup_median_gpu_fpga_to_cpu_fpga, speedup_median_gpu_fpga_to_cpu_fpga])      
        max_speedup_median_cpu_gpu_to_cpu = np.amax([max_speedup_median_cpu_gpu_to_cpu, speedup_median_cpu_gpu_to_cpu])
        max_speedup_median_cpu_fpga_to_cpu = np.amax([max_speedup_median_cpu_fpga_to_cpu, speedup_median_cpu_fpga_to_cpu])
        max_speedup_median_gpu_fpga_to_cpu = np.amax([max_speedup_median_gpu_fpga_to_cpu, speedup_median_gpu_fpga_to_cpu])
        max_speedup_median_gpu_fpga_to_cpu_fpga = np.amax([max_speedup_median_gpu_fpga_to_cpu_fpga, speedup_median_gpu_fpga_to_cpu_fpga])
        global min_speedup_p95_cpu_gpu_to_cpu, min_speedup_p95_cpu_fpga_to_cpu, min_speedup_p95_gpu_fpga_to_cpu, min_speedup_p95_gpu_fpga_to_cpu_fpga
        global max_speedup_p95_cpu_gpu_to_cpu, max_speedup_p95_cpu_fpga_to_cpu, max_speedup_p95_gpu_fpga_to_cpu, max_speedup_p95_gpu_fpga_to_cpu_fpga
        min_speedup_p95_cpu_gpu_to_cpu = np.amin([min_speedup_p95_cpu_gpu_to_cpu, speedup_p95_cpu_gpu_to_cpu])
        min_speedup_p95_cpu_fpga_to_cpu = np.amin([min_speedup_p95_cpu_fpga_to_cpu, speedup_p95_cpu_fpga_to_cpu])
        min_speedup_p95_gpu_fpga_to_cpu = np.amin([min_speedup_p95_gpu_fpga_to_cpu, speedup_p95_gpu_fpga_to_cpu])
        min_speedup_p95_gpu_fpga_to_cpu_fpga = np.amin([min_speedup_p95_gpu_fpga_to_cpu_fpga, speedup_p95_gpu_fpga_to_cpu_fpga])
        max_speedup_p95_cpu_gpu_to_cpu = np.amax([max_speedup_p95_cpu_gpu_to_cpu, speedup_p95_cpu_gpu_to_cpu])
        max_speedup_p95_cpu_fpga_to_cpu = np.amax([max_speedup_p95_cpu_fpga_to_cpu, speedup_p95_cpu_fpga_to_cpu])
        max_speedup_p95_gpu_fpga_to_cpu = np.amax([max_speedup_p95_gpu_fpga_to_cpu, speedup_p95_gpu_fpga_to_cpu])
        max_speedup_p95_gpu_fpga_to_cpu_fpga = np.amax([max_speedup_p95_gpu_fpga_to_cpu_fpga, speedup_p95_gpu_fpga_to_cpu_fpga])


    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])

    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]


    ax.scatter(inds, medians, marker='o', color=colors['median_bubble'], s=3, zorder=3)
    # ax.hlines(medians, inds - 0.1, inds + 0.1, color=colors['box_plot'], linestyle='-', lw=1)
    # ax.vlines(inds, quartile1, quartile3, color=colors['box_plot'], linestyle='-', lw=5)
    # ax.vlines(inds, whiskers_min, whiskers_max, color=colors['box_plot'], linestyle='-', lw=1)

    set_axis_style(ax, inds, label_positions, labels)

    legend_left = 0.0 if dbname != 'RALM-L1000M' else -0.02
    ax.legend(architectures, loc=(legend_left, 1.02), ncol=4, frameon=False, fontsize=legend_font)
    print_dbname = dbname
    print_dbname = print_dbname.replace('SIFT1000M', 'SIFT')
    print_dbname = print_dbname.replace('Deep1000M', 'Deep')
    print_dbname = print_dbname.replace('RALM-S1000M', 'SYN-512')
    print_dbname = print_dbname.replace('RALM-L1000M', 'SYN-1024')
    ax.text(1, ax.get_ylim()[1] / 1.5, f"Dataset: {print_dbname}", fontsize=legend_font, verticalalignment='top', horizontalalignment='left')

    # plt.rcParams.update({'figure.autolayout': True})
    fig.tight_layout()
    plt.savefig(f'./images/latency_{dbname}.png', transparent=False, dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    for dbname in dbname_list:
        #plot_seaborn(dbname)
        plot_matplotlib(dbname)

    print("==== Summary ====")
    print("Median speedup: CPU-GPU/CPU: {:.2f}~{:.2f}\tCPU-FPGA/CPU: {:.2f}~{:.2f}\tGPU-FPGA/CPU: {:.2f}~{:.2f}\tGPU-FPGA/CPU-FPGA: {:.2f}~{:.2f}".format(
        min_speedup_median_cpu_gpu_to_cpu, max_speedup_median_cpu_gpu_to_cpu, 
        min_speedup_median_cpu_fpga_to_cpu, max_speedup_median_cpu_fpga_to_cpu, 
        min_speedup_median_gpu_fpga_to_cpu, max_speedup_median_gpu_fpga_to_cpu, 
        min_speedup_median_gpu_fpga_to_cpu_fpga, max_speedup_median_gpu_fpga_to_cpu_fpga))
    print("95th percentile speedup: CPU-GPU/CPU: {:.2f}~{:.2f}\tCPU-FPGA/CPU: {:.2f}~{:.2f}\tGPU-FPGA/CPU: {:.2f}~{:.2f}\tGPU-FPGA/CPU-FPGA: {:.2f}~{:.2f}".format(
        min_speedup_p95_cpu_gpu_to_cpu, max_speedup_p95_cpu_gpu_to_cpu,
        min_speedup_p95_cpu_fpga_to_cpu, max_speedup_p95_cpu_fpga_to_cpu,
        min_speedup_p95_gpu_fpga_to_cpu, max_speedup_p95_gpu_fpga_to_cpu,
        min_speedup_p95_gpu_fpga_to_cpu_fpga, max_speedup_p95_gpu_fpga_to_cpu_fpga))
        