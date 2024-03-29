"""
Calculate the least DB size, given a certain LUT construction performance

The result is related to the following parameters:
	D: dimension of the vectors
	M: number of subvectors
	nlist: number of cells in the IVF
	N_LUC_CONTRUCTION_PE: number of LUT construction PEs
"""

import numpy as np

N_channels = 4

D = 512
M = 32
AXI_BYTES = 64
nlist = 32768

N_LUC_CONTRUCTION_PE = 8

if D/M <= 8:
    II = 1
elif D/M <= 16:
    II = 2
elif D/M <= 32:
    II = 4

# per LUT construction
num_outputs = 256 * M
cycles_construction = num_outputs / N_LUC_CONTRUCTION_PE * II

vec_per_round = N_channels * (AXI_BYTES / M)
num_least_vectors_per_list = cycles_construction * vec_per_round 
num_least_vectors_in_db = num_least_vectors_per_list * nlist

print(f"cycles_construction = {cycles_construction}")
print(f"num_least_vectors_per_list = {num_least_vectors_per_list}")
print(f"num_least_vectors_in_db = {num_least_vectors_in_db} = {num_least_vectors_in_db / 1e9} billions")