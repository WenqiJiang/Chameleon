import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def get_statistics(y: np.ndarray, baseline_mean: Optional[float]=None) -> List[float]:
    # manage the table index in this function as it has to match the order of the columns
    table_index = ['mean', 'speedup', 'min', '5th-percentile', '25th-percentile', 'median', '75th-percentile', '95th-percentile', 'max', 'range (max-min)']

    y_mean = np.mean(y)
    y_min = np.min(y)
    y_max = np.max(y)
    if baseline_mean is None:
        baseline_mean = y_mean
    y_speedup = baseline_mean / y_mean
    y_range = y_max - y_min
    percentile5, percentile25, median, percentile75, percentile95 = np.percentile(np.sort(y.flatten()), [5, 25, 50, 75, 95])

    return [y_mean, y_speedup, y_min, percentile5, percentile25, median, percentile75, percentile95, y_max, y_range], table_index

def clean_column_for_latex(column: List[float]):
    return list(map('{:.2f}'.format, column))

def append_column_to_table(table: dict, column: List[str], column_name: str):
    table[column_name] = column

def save_table_as_latex(table: dict, index: list[str], filename: str):
    df = pd.DataFrame(table, index=index)
    with open(f'tables/{filename}.tex', 'w') as f:
        f.write(df.to_latex())

def extract_data_from_result_dict(results_dict, model_name, architecture, retrieval_interval, batch_size):
    y_latency_ms = results_dict[model_name][architecture][retrieval_interval][batch_size]['latency_ms']
    # return the average latency over the three series of measurements per step
    return y_latency_ms.mean(axis=0)

latency_dict_baseline = load_obj('./performance_results_archive/', 'RALM_latency_throughput_CPU_GPU')
latency_dict_FPGA = load_obj('./performance_results_archive/', 'RALM_latency_throughput_FPGA')
# print(throughput_dict)


def create_table_and_save(model_name, retrieval_intervals):
    #model_name = 'Dec-S'
    retrieval_interval = retrieval_intervals[0]
    batch_size = 1

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

    column_headers = []

    columns = []

    for architecture in [CPU_architecture, FPGA_architecture]:
        if 'FPGA' in architecture:
            latency_dict = latency_dict_FPGA
        else: 
            latency_dict = latency_dict_baseline

        for retrieval_interval in retrieval_intervals:
            column_headers.append(f"{architecture}")
            #column_headers.append(f"{architecture} - {retrieval_interval}")
            columns.append(extract_data_from_result_dict(latency_dict, model_name, architecture, retrieval_interval, batch_size))
            

    # asume the first combination of architecture and retrieval interval is the baseline
    baseline_header = column_headers[0]
    baseline_statistics, table_index = get_statistics(columns[0])
    baseline_mean = baseline_statistics[0]
    baseline_column = clean_column_for_latex(baseline_statistics)

    table = {}
    append_column_to_table(table, baseline_column, baseline_header)

    for header, column in zip(column_headers[1:], columns[1:]):
        statistic, table_index = get_statistics(column, baseline_mean)
        final_column = clean_column_for_latex(statistic)
        append_column_to_table(table, final_column, header)


    save_table_as_latex(table, table_index, f'statistics_table_{model_name}_{retrieval_interval}')

if __name__ == '__main__':
    create_table_and_save("Dec-S", retrieval_intervals=[1])
    for retrieval_interval in [8, 64, 512]:
        create_table_and_save("EncDec-S", retrieval_intervals=[retrieval_interval])
    create_table_and_save("Dec-L", retrieval_intervals=[1])
    for retrieval_interval in [8, 64, 512]:
        create_table_and_save("EncDec-L", retrieval_intervals=[retrieval_interval])