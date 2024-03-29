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



def plot_and_save(model_names, plotname='small'):

    x_labels = []
    ys_baseline = []
    ys_FPGA = []

    label_font = 10
    markersize = 8
    tick_font = 8
    legend_font = 10

    for model_name in model_names:

        if model_name == 'Dec-S': 
            CPU_architecture = '8CPU'
            FPGA_architecture = '1FPGA-1GPU'
            retrieval_intervals = [1]
            batch_size = 64
        elif model_name == 'EncDec-S':
            CPU_architecture = '8CPU'
            FPGA_architecture = '1FPGA-1GPU'
            retrieval_intervals = [8, 64, 512]
            batch_size = 64
        elif model_name == 'Dec-L':
            CPU_architecture = '16CPU'
            FPGA_architecture = '2FPGA-1GPU'
            retrieval_intervals = [1]
            batch_size = 8
        elif  model_name == 'EncDec-L':
            CPU_architecture = '16CPU'
            FPGA_architecture = '2FPGA-1GPU'
            retrieval_intervals = [8, 64, 512]
            batch_size = 8
            
        if CPU_architecture == '8CPU':
            color_A = default_colors[0]
        elif CPU_architecture == '16CPU':
            color_A = default_colors[2]
        if FPGA_architecture == '1FPGA-1GPU':
            color_B = default_colors[1]
        elif FPGA_architecture == '2FPGA-1GPU':
            color_B = default_colors[3]
            
        for retrieval_interval in retrieval_intervals:
            x_label = f'{model_name}\nInterval={retrieval_interval}'
            y_baseline = latency_dict_baseline[model_name][CPU_architecture][retrieval_interval][batch_size]['throughput_tokens_per_sec']
            y_FPGA = latency_dict_FPGA[model_name][FPGA_architecture][retrieval_interval][batch_size]['throughput_tokens_per_sec']
            x_labels.append(x_label)
            ys_baseline.append(y_baseline)
            ys_FPGA.append(y_FPGA)
            print("Model: {}, retrieval_interval: {}, batch_size: {}, speedup: {:.2f}".format(
                model_name, retrieval_interval, batch_size, y_FPGA.mean() / y_baseline.mean()))

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

    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(6, 2))
    rects1  = ax.bar(x - width/2, y_baseline_means, width, color=color_A)#, label='Men')
    rects2 = ax.bar(x + width/2, y_FPGA_means, width, color=color_B)#, label='Women')
    ax.errorbar(x - width/2, y_baseline_means, yerr=yerr_baseline, fmt='none', color='black', capsize=3)
    ax.errorbar(x + width/2, y_FPGA_means, yerr=yerr_FPGA, fmt='none', color='black', capsize=3)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Throughput\n(tokens / sec)', fontsize=label_font)
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=label_font)
    ax.legend([rects1, rects2], [CPU_architecture + '-1GPU', FPGA_architecture], loc=(0.2, 1.02), ncol=2, \
    facecolor='white', framealpha=1, frameon=False, fontsize=legend_font)


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
                        xytext=(text_x, 4),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0, fontsize=tick_font)


    autolabel(rects1, align='left')
    autolabel(rects2, align='right')
    ax.tick_params(axis='y', which='major', labelsize=tick_font)
    ax.set_ylim([0, 1.3 * np.amax(ys_FPGA)])

    # plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'figure.autolayout': True})
    fig.tight_layout()
    plt.savefig(f'./images/ralm_throughput_{plotname}.png', transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()

if __name__ == '__main__':

    model_names_small = ['Dec-S', 'EncDec-S'] 
    model_names_large = ['Dec-L', 'EncDec-L']

    plot_and_save(model_names_small, 'small')
    plot_and_save(model_names_large, 'large')