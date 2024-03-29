# Specification

## P & R issues

For `accelerator_D512_M32_batch` and `accelerator_D1024_M64_batch`, we use extended FIFO for LUT construction, because the LUT construction in those repositories are not fast enough (8 and 16 PEs, respectively, both using II=1), see `LUT_construction.hpp` for details. While for SIFT and Deep with PQ=16, the LUT construction performance can match the line rate (16 PEs, II-1), so no need to add that extra FIFO.

Solved: `accelerator_D512_M32_batch` does not went through P & R. So we downgrade the number LUT_CONSTR_SUB_PE_NUM from 16 to 8, and choose the `Performance_BalanceSLLs` to fix the P & R issue. 

Unsolved: `accelerator_D512_M64_batch`. I tried three P&R strategies, but all failed. Maybe changing LUT_CONSTR_SUB_PE_NUM from 16 to 8 would help.

## Streaming Processing Kernels

The only difference between SIFT 32 and Deep 32 kernels is the constant definition:

```
(base) wejiang@hacc-build-01:/mnt/scratch/wenqi/FPGA-PQ-disaggregated-memory/kernel/user_krnl$ diff -r accelerator_SIFT_M32/src/hls/ 
accelerator_Deep_M32/src/hls/
Only in accelerator_Deep_M32/src/hls/: accelerator_Deep_M32.cpp
Only in accelerator_SIFT_M32/src/hls/: accelerator_SIFT_M32.cpp
diff -r accelerator_SIFT_M32/src/hls/constants.hpp accelerator_Deep_M32/src/hls/constants.hpp
10c10
< #define D 128
---
> #define D 96
```

The GNN 64 and SBERT 64 uses different LUT constructor. Specifically, for the 32-byte PQ code version, the number of LUT construction PE = M. While in the 64-byte version, the LUT constructor PE number can be less than M. 

Nevertheless, each constructor PE can still construct the one value of the distance LUT (pipeline II=1): 

```
            for (int k = 0; k < LUT_ENTRY_NUM; k++) {

                for (int m_per_PE = 0; m_per_PE < (M / LUT_CONSTR_SUB_PE_NUM); m_per_PE++) {
#pragma HLS pipeline II=1

                    // the L1 diff between sub_residual_vector annd sub-quantizers
                    float L1_dist[D/M];
#pragma HLS array_partition variable=L1_dist complete

                        for (int j = m_per_PE * (D / M); j < (m_per_PE + 1) * (D / M); j++) {
#pragma HLS UNROLL
                            L1_dist[j] = sub_residual_vector[j] - sub_product_quantizer[k][j];
                        }

                    // square distance
                    float L2_dist = square_sum<D / M>(L1_dist);
                    
                    s_partial_distance_LUT.write(L2_dist);
                }
            }
```