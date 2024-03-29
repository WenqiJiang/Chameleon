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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# plt.style.use('ggplot')
plt.style.use('seaborn-pastel') 

def get_default_colors():

  default_colors = []
  for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
      default_colors.append(color["color"])
      print(color["color"], type(color["color"]))

  return default_colors

default_colors = get_default_colors()
# print(default_colors[0], type(default_colors[0]))

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

latency_dict_baseline = load_obj('./performance_results_archive/', 'RALM_latency_throughput_CPU_GPU')
latency_dict_FPGA = load_obj('./performance_results_archive/', 'RALM_latency_throughput_FPGA')
# print(throughput_dict)


def plot_and_save(model_name, retrieval_interval, line=True, scatter=False, min_max=False, quartiles=False, density_plot=False):

    label_font = 11
    markersize = 8
    tick_font = 9
    legend_font = 10

    if model_name == 'Dec-S': 
        CPU_architecture = '8CPU'
        FPGA_architecture = '1FPGA-1GPU'
    elif model_name == 'EncDec-S':
        CPU_architecture = '8CPU'
        FPGA_architecture = '1FPGA-1GPU'
    elif model_name == 'Dec-L':
        CPU_architecture = '16CPU'
        FPGA_architecture = '2FPGA-1GPU'
    elif  model_name == 'EncDec-L':
        CPU_architecture = '16CPU'
        FPGA_architecture = '2FPGA-1GPU'

    if CPU_architecture == '8CPU':
        color_A = default_colors[0]
    elif CPU_architecture == '16CPU':
        color_A = default_colors[2]
    if FPGA_architecture == '1FPGA-1GPU':
        color_B = default_colors[1]
    elif FPGA_architecture == '2FPGA-1GPU':
        color_B = default_colors[3]
    x = np.arange(0, 512)

    batch_size = 1
    y_baseline_latency_ms = latency_dict_baseline[model_name][CPU_architecture][retrieval_interval][batch_size]['latency_ms']
    y_FPGA_latency_ms = latency_dict_FPGA[model_name][FPGA_architecture][retrieval_interval][batch_size]['latency_ms']
    # print(y_baseline_latency_ms.shape)
    # print(y_FPGA_latency_ms.shape)
    y_baseline_latency_ms_average = np.mean(y_baseline_latency_ms, axis=0)
    y_FPGA_latency_ms_average = np.mean(y_FPGA_latency_ms, axis=0)

    speedup_e2e = y_baseline_latency_ms_average.mean() / y_FPGA_latency_ms_average.mean()
    speedup_retrieval_step = y_baseline_latency_ms_average[::retrieval_interval] / y_FPGA_latency_ms_average[::retrieval_interval]
    print("Model: {}, retrieval_interval: {}, speedup (e2e): {:.2f}, speedup (retrieval step): {:.2f} ~ {:.2f}".format(
        model_name, retrieval_interval, speedup_e2e, speedup_retrieval_step.min(), speedup_retrieval_step.max()))

    # Rewrite the CPU architecture, add GPU
    CPU_architecture = CPU_architecture + '-1GPU'

    if density_plot:
        fig, (ax, ax_histogram) = plt.subplots(1, 2, figsize=(6, 2), width_ratios=[4, 1])

    else:
        fig, ax = plt.subplots(figsize=(6, 2))

    if line:
        ax.plot(x, y_baseline_latency_ms_average, linewidth=2, label=CPU_architecture, color=color_A)
        ax.plot(x, y_FPGA_latency_ms_average, linewidth=2, label=FPGA_architecture, color=color_B)
    if scatter:
        ax.scatter(x, y_baseline_latency_ms_average, marker=".", linewidth=0, label=CPU_architecture, color=color_A)
        ax.scatter(x, y_FPGA_latency_ms_average, marker=".", linewidth=0, label=FPGA_architecture, color=color_B)

    len_x = x.shape[0]

    y_baseline_latency_ms_min = np.min(y_baseline_latency_ms_average)
    y_baseline_latency_ms_max = np.max(y_baseline_latency_ms_average)
    baseline_quartile1, baseline_medians, baseline_quartile3 = np.percentile(np.sort(y_baseline_latency_ms_average.flatten()), [25, 50, 75])
    y_lines_baseline = []

    if min_max:
        y_lines_baseline.append(y_baseline_latency_ms_min)
    if quartiles:
        y_lines_baseline.append(baseline_quartile1)
        y_lines_baseline.append(baseline_medians)
        y_lines_baseline.append(baseline_quartile3)
    if min_max:
        y_lines_baseline.append(y_baseline_latency_ms_max)

    ax.hlines(y_lines_baseline, len(y_lines_baseline)*[0], len(y_lines_baseline)*[len_x], color='k', linestyle='-', lw=1)

    y_FPGA_latency_ms_min = np.min(y_FPGA_latency_ms)
    y_FPGA_latency_ms_max = np.max(y_FPGA_latency_ms)

    #means = np.mean(data, axis=1)

    if density_plot:
        ax_histogram.hist([y_baseline_latency_ms_average, y_FPGA_latency_ms_average], bins=50, orientation='horizontal', alpha=0.5, log=False, density=True, histtype='stepfilled', color=[color_A, color_B]) 
        ax_histogram.set_xlabel('Prob. density', fontsize=label_font, loc='right')
        ax_histogram.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax_histogram.tick_params(axis='x', labelsize=tick_font)
        ax_histogram.set_xscale("log")
    
    ax.set_ylabel('Latency (ms)', fontsize=label_font)
    ax.set_xlabel('Time (token) steps', fontsize=label_font)
    ax.tick_params(labelsize=tick_font)

    ax.legend([CPU_architecture, FPGA_architecture], loc=(0, 1.01), ncol=2, frameon=False, fontsize=legend_font)

    # if density_plot:
    #     model_x = 1.35 * 512
    # else:
    #     model_x = 1.0 * 512
    # ax.text(model_x, ax.get_ylim()[1] * 1.02, f"Model: {model_name}, Interval: {retrieval_interval}", fontsize=label_font, verticalalignment='bottom', horizontalalignment='right')
    if density_plot:
        ax_histogram.set_title(f"{model_name}, Interval={retrieval_interval}", fontsize=label_font, loc='right')
    else:
        ax.set_title(f"{model_name}, Interval={retrieval_interval}", fontsize=label_font, loc='right')

    # ax.set_ylim([0, 100])
    plt.rcParams.update({'figure.autolayout': True})
    fig.tight_layout()
    plt.savefig(f'./images/ralm_latency_{model_name}_{retrieval_interval}.png', dpi=500)# pad_inches=0)

if __name__ == '__main__':
    
    line = True
    scatter = False
    min_max = False
    quartiles = False
    density_plot = True
    
    for retrieval_interval in [1]:
        plot_and_save("Dec-S", retrieval_interval, line=line, scatter=scatter, min_max=min_max, quartiles=quartiles, density_plot=density_plot)

    for retrieval_interval in [8, 64, 512]:
        plot_and_save("EncDec-S", retrieval_interval, line=line, scatter=scatter, min_max=min_max, quartiles=quartiles, density_plot=density_plot)

    for retrieval_interval in [1]:
        plot_and_save("Dec-L", retrieval_interval, line=line, scatter=scatter, min_max=min_max, quartiles=quartiles, density_plot=density_plot)

    for retrieval_interval in [8, 64, 512]:
        plot_and_save("EncDec-L", retrieval_interval, line=line, scatter=scatter, min_max=min_max, quartiles=quartiles, density_plot=density_plot)