"""
The RALM performance is stored in a hierachical dictionary: 
    d_perf = d[model_name][architecture][retrieval_interval][batch_size]
        (we assume [model_name][index][k][nprobe] are tied to a certain model)
        d_perf['latency_ms'] = [latency_array_1 (batch 1), latency_array_2 (batch 2), ...],
            np array, shape = (batch_size, seq_len)
        d_perf['throughput_tokens_per_sec'] = [throughput_1, throughput_2, ...],
            np array, shape = (batch_size,)
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
sns.set_palette("pastel")

### Note: For violin graph, a single violin's data must be in the same column
###   e.g., given 3 violin plots, each with 100 points, the shape of the array
###   should be (100, 3), where the first column is for the first violin and so forth
# fake up some data


# Wenqi: flatten the table to a table. It's a dictionary with the key as schema.
#   The value of each key is an array.
# label category data
# xxx   xxx      xxx
# yyy   yyy      yyy


colors = {'8CPU': 'tab:blue',
          '1FPGA-1GPU': 'tab:orange',
          'box_plot': 'k',
          'mean_bubble': 'white',
          }

batch_size = 1
model_name_list = ['Dec-S', 'EncDec-S',]# 'Dec-L', 'EncDec-L']

def get_architectures_and_retrieval_intervals(model_name):
    if model_name == 'Dec-S': 
        architectures = ['8CPU', '1FPGA-1GPU']
        retrieval_intervals = [1]
    elif model_name == 'EncDec-S':
        architectures = ['8CPU', '1FPGA-1GPU']
        retrieval_intervals = [8, 64, 512]
    elif model_name == 'Dec-L':
        architectures = ['16CPU','2FPGA-1GPU']
        retrieval_intervals = [1]
    elif  model_name == 'EncDec-L':
        architectures = ['16CPU', '2FPGA-1GPU']
        retrieval_intervals = [8, 64, 512]
    
    return architectures, retrieval_intervals


def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)
        
def plot_seaborn(model_name):
    architectures, retrieval_intervals = get_architectures_and_retrieval_intervals(model_name)

    d = {}
    d['label'] = [] # batch size
    d['data'] = []
    d['category'] = [] # architecture

    latency_dict_baseline = load_obj('./performance_results_archive/', 'RALM_latency_throughput_CPU_GPU')
    latency_dict_FPGA = load_obj('./performance_results_archive/', 'RALM_latency_throughput_FPGA')

    for architecture in architectures:
        for retrieval_interval in retrieval_intervals:
            if 'FPGA' in architecture:
                latency_array = latency_dict_FPGA[model_name][architecture][retrieval_interval][batch_size]['latency_ms']
            else:
                latency_array = latency_dict_baseline[model_name][architecture][retrieval_interval][batch_size]['latency_ms']
            # latency_array = latency_array.flatten() # originally (batch_size, seq_len)
            # latency_array = np.median(latency_array, axis=0) # (seq_len,)
            latency_array = np.average(latency_array, axis=0) # (seq_len,)
            for latency in latency_array:
                d['label'].append(retrieval_interval)
                d['data'].append(latency)
                d['category'].append(architecture)
            
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

    x_tick_labels = [model_name + f", interval={retrieval_interval}" for retrieval_interval in retrieval_intervals]
    ax.set_xticklabels(x_tick_labels, fontsize=tick_font)
    # ax.set_yticklabels(y_tick_labels)
    plt.yticks(fontsize=tick_font)
    # # ax.ticklabel_format(axis='both', style='sci')
    ax.set_yscale("log")
    loc = (0.02,1.02)#"upper left" # "best"
    ax.legend(loc=loc, ncol=3, fontsize=legend_font, frameon=False)

    ax.tick_params(length=0, top=False, bottom=False, left=False, right=False, 
        labelleft=True, labelbottom=True, labelsize=12)
    # ax.set_xlabel('Model & Interval', fontsize=label_font, labelpad=10)
    ax.set_ylabel('Latency (ms)', fontsize=label_font, labelpad=10)
    # ax.set_title(f'{model_name}', fontsize=label_font, y=1.35)
    # ax.text(2.5, 5, f'{model_name}', fontsize=label_font)
    # plt.text(2, len(y_tick_labels) + 2, "Linear Heatmap", fontsize=16)

    plt.savefig(f'./images/ralm_latency_{model_name}.png', transparent=False, dpi=200, bbox_inches="tight")

    # plt.show()

def plot_matplotlib(model_name):
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
        ax.set_ylabel('Latency (ms)')
        ax.set_xlabel('Retrieval Interval')

    architectures, retrieval_intervals = get_architectures_and_retrieval_intervals(model_name)

    latency_dict_baseline = load_obj('./performance_results_archive/', 'RALM_latency_throughput_CPU_GPU')
    latency_dict_FPGA = load_obj('./performance_results_archive/', 'RALM_latency_throughput_FPGA')

    architecture_labels = []
    retrieval_interval_labels = []

    alpha_violin_plot = 0.6
    relative_distance_between_plot_groups = 1.5

    data = []
    inds = [] # x positions of the violin charts
    curr_x_position = 1

    labels = []
    label_positions = []
    label_position = 0

    for retrieval_interval in retrieval_intervals:
        for architecture in architectures:
            if 'FPGA' in architecture:
                latency_array = latency_dict_FPGA[model_name][architecture][retrieval_interval][batch_size]['latency_ms']
            else:
                latency_array = latency_dict_baseline[model_name][architecture][retrieval_interval][batch_size]['latency_ms']
            architecture_labels.append(architecture)
            retrieval_interval_labels.append(retrieval_interval)
            data.append(np.sort(latency_array.flatten()))
            inds.append(curr_x_position)
            label_position += curr_x_position
            curr_x_position += 1
        curr_x_position += relative_distance_between_plot_groups - 1
        label_position /= len(architectures)
        labels.append(f"{retrieval_interval}")
        label_positions.append(label_position)
        label_position = 0
    
    inds = np.array(inds)


    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.set_yscale('log')

    parts = ax.violinplot(data, inds, vert=True, showmeans=False, showmedians=False, showextrema=False, widths=1.8/len(architectures)) 

    for pc, architecture_label in zip(parts['bodies'], architecture_labels):
        pc.set_facecolor(colors[architecture_label])
        pc.set_edgecolor('black')
        pc.set_linewidth(0.75)
        pc.set_alpha(alpha_violin_plot)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    means = np.mean(data, axis=1)

    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])

    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]


    ax.scatter(inds, means, marker='o', color=colors['mean_bubble'], s=3, zorder=3)
    ax.hlines(medians, inds - 0.1, inds + 0.1, color=colors['box_plot'], linestyle='-', lw=1)
    ax.vlines(inds, quartile1, quartile3, color=colors['box_plot'], linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color=colors['box_plot'], linestyle='-', lw=1)

    set_axis_style(ax, inds, label_positions, labels)

    ax.legend(architectures, bbox_to_anchor=(0, 1.04, 1, 0.2), loc='lower left', ncol=len(architectures), frameon=False, prop={'size': 10})
    ax.set_title(f"Model: {model_name}", loc='right')

    plt.savefig(f'./images/ralm_latency_{model_name}.png', transparent=False, dpi=500, bbox_inches="tight")



if __name__ == "__main__":
    for model_name in model_name_list:
        plot_matplotlib(model_name)