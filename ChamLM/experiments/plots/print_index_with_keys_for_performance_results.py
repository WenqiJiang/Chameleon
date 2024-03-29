import os
import pickle
import numpy as np
import pandas as pd

columns = ['model', 'architecture', 'retrieval_interval', 'batch_size', 'measurement', 'data']
keys = [None for _ in columns[:-1]]
records = []

def recursive_key_print(dictionary, indent=0):
    for key, value in dictionary.items():
        keys[indent] = key
        if isinstance(value, np.ndarray):
            print('\t' * indent + str(key) + " " + str(value.shape))
            record = {column: key for column, key in zip(columns, keys)}
            record['data'] = value
            records.append(record)
        else:
            print('\t' * indent + str(key))
        if isinstance(value, dict):
            recursive_key_print(value, indent+1)
    
    return pd.DataFrame(records)


def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

latency_dict_baseline = load_obj('./performance_results_archive/', 'RALM_latency_throughput_CPU_GPU')

records_df = recursive_key_print(latency_dict_baseline)
print(records_df)