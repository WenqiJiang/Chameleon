import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import pandas as pd

plt.style.use('seaborn-colorblind') 

dirc = './performance_results_archive/'
file1 = 'RALM_latency_throughput_CPU_GPU'
file2 = 'RALM_latency_throughput_FPGA'

retrieval_interval = 8
model_name = 'EncDec-S' #'Dec-S'
CPU_architecture = '8CPU'
FPGA_architecture = '1FPGA-1GPU'
batch_size = 1

def load_data():
    def load_obj(dirc, name):
        with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
            return pickle.load(f)

    latency_dict_baseline = load_obj(dirc, file1)
    latency_dict_FPGA = load_obj(dirc, file2)


    y1 = latency_dict_baseline[model_name][CPU_architecture][retrieval_interval][batch_size]['latency_ms']
    y2 = latency_dict_FPGA[model_name][FPGA_architecture][retrieval_interval][batch_size]['latency_ms']

    return y1, y2


def get_flat_data_as_df(y1, y2):
    y_1 = y1.flatten()[0:75]
    y_2 = y2.flatten()[0:75]

    d = {}
    d['label'] = [] # batch size
    d['data'] = []
    d['category'] = [] # architecture

    for i in range(0, len(y_1)):
        d['label'].append(batch_size)
        d['data'].append(y_1[i])
        d['category'].append('CPU')

    for i in range(0, len(y_2)):
        d['label'].append(batch_size)
        d['data'].append(y_2[i])
        d['category'].append('FPGA')

    return pd.DataFrame(data=d)

def violin_cat_plot():
    y_baseline_latency_ms, y_FPGA_latency_ms = load_data()
    df = get_flat_data_as_df(y_baseline_latency_ms, y_FPGA_latency_ms)

    plt.legend(prop={'size': 12})
    sns.violinplot(data=df, scale='width', inner=None, x="label", y="data", hue="category", color=".9")
    sns.swarmplot(data=df, x="label", y="data", hue="category", dodge=True, size=2)

    plt.savefig(f'./images/test_plot.png', dpi=500)

def density_plot(show_hist=True, show_kde=True):
    y_baseline_latency_ms, y_FPGA_latency_ms = load_data()
    df = get_flat_data_as_df(y_baseline_latency_ms, y_FPGA_latency_ms)

    df_1 = df[df.category == 'CPU']
    sns.distplot(df_1['data'], hist = show_hist, kde = show_kde, label='8CPU')
    df_2 = df[df.category == 'FPGA']
    sns.distplot(df_2['data'], hist = show_hist, kde = show_kde, label='1FPGA-1GPU')
    # Plot formatting
    plt.legend(prop={'size': 12})
    plt.title('Latency Distribution')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Density')

    plt.savefig(f'./images/test_plot.png', dpi=500)

def line_plot(cpu=True, fpga=True, selected_interval_start=0, selected_interval_end=50):
    y_baseline_latency_ms, y_FPGA_latency_ms = load_data()

    x = np.arange(selected_interval_start, selected_interval_end)

    fig, ax = plt.subplots()

    ax.set_ylabel('Per-step Latency (ms)')
    ax.set_xlabel('Time Step')

    if cpu and fpga:
        y_1 = np.average(y_baseline_latency_ms, axis=0)
        y_2 = np.average(y_FPGA_latency_ms, axis=0)
        ax.plot(x, y_1[selected_interval_start:selected_interval_end], color='green', linewidth=1) 
        ax.plot(x, y_2[selected_interval_start:selected_interval_end], color='blue', linewidth=1) 
        plt.savefig(f'./images/test_plot.png', dpi=500)
        return

    elif cpu:
        ys = y_baseline_latency_ms
    elif fpga:
        ys = y_FPGA_latency_ms

    print(ys.shape)

    ax.plot(x, ys[0,selected_interval_start:selected_interval_end], color='green', linewidth=1) 
    ax.plot(x, ys[1,selected_interval_start:selected_interval_end], color='blue', linewidth=1) 
    ax.plot(x, ys[2,selected_interval_start:selected_interval_end], color='red', linewidth=1) 
    plt.savefig(f'./images/test_plot.png', dpi=500)

    #y = np.max(y_baseline_latency_ms, axis=0) - np.min(y_baseline_latency_ms, axis=0)
    #ax.plot(x, y, color='green', linewidth=1) 

def violin_matplotlib(showmeans=True, showmedians=False, showextrema=True):
    y_baseline_latency_ms, y_FPGA_latency_ms = load_data()
    y_1 = y_baseline_latency_ms.flatten()
    y_2 = y_FPGA_latency_ms.flatten()

    fig, ax = plt.subplots()
    ax.violinplot([y_1, y_2], vert=True, showmeans=showmeans, showmedians=showmedians, showextrema=showextrema)
    plt.savefig(f'./images/test_plot.png', dpi=500)

def customized_violin_matplotlib():
    # https://matplotlib.org/stable/gallery/statistics/customized_violin.html
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value


    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)

        ax.set_ylabel('Latency (ms)')
        ax.set_xlabel('Architecture')

    y_baseline_latency_ms, y_FPGA_latency_ms = load_data()
    y_1 = np.sort(y_baseline_latency_ms.flatten())
    y_2 = np.sort(y_FPGA_latency_ms.flatten())
    data = [y_1, y_2]

    fig, ax = plt.subplots()
    parts = ax.violinplot(data, vert=True, showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        #pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    means = np.mean(data, axis=1)

    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])

    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    annotations = ["inference only", "inference +\nretrieval on FPGA", "inference +\nretrieval on CPU"]
    ax.annotate(annotations[0], xy=(1, 15), xytext=(1.30, 19), fontsize=10)  # manually added for annotation purposes
    ax.annotate(annotations[1], xy=(1, 25), xytext=(2.10, 26), fontsize=10)  # manually added for annotation purposes
    ax.annotate(annotations[2], xy=(1, 50), xytext=(1.10, 56), fontsize=10)  # manually added for annotation purposes
    inds = np.arange(1, len(medians) + 1)
    #ax.scatter(inds, medians, marker='x', color='black', s=30, zorder=3)
    ax.scatter(inds, means, marker='o', color='white', s=10, zorder=3)
    #ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    #ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    ax.hlines([medians[1], 25, 55], [0.5, 1.5, 0.5], [2.5, 2.5, 1.5], color='k', linestyle='-', lw=1) # manually added for annotation purposes
    # set style for the axes
    labels = ['CPU', 'FPGA']
    set_axis_style(ax, labels)
    plt.savefig(f'./images/test_plot.png', dpi=500)


def overlapping_histograms(log=True):
    y_baseline_latency_ms, y_FPGA_latency_ms = load_data()
    y_1 = y_baseline_latency_ms.flatten()
    y_2 = y_FPGA_latency_ms.flatten()

    #y_min = np.amin([y_1, y_2])
    y_min = 0
    y_max = np.amax([y_1, y_2])

    bins = np.linspace(y_min, y_max, 200)

    fig, ax = plt.subplots()

    ax.hist(y_1, bins, alpha=0.6, label='CPU', log=True)
    ax.hist(y_2, bins, alpha=0.6, label='FPGA', log=True)

    plt.legend(prop={'size': 12})
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Latency Distribution')

    plt.savefig(f'./images/test_plot.png', dpi=500)

if __name__ == '__main__':
    #violin_cat_plot()
    #density_plot(show_hist=True, show_kde=False)
    #line_plot(cpu=True, fpga=True, selected_interval_start=0, selected_interval_end=50)
    #violin_matplotlib(showmeans=True, showmedians=True, showextrema=True)
    customized_violin_matplotlib()
    #overlapping_histograms(log=True)