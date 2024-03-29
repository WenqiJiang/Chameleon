# M=64 specification

(1) LUT construction PE different than SIFT and Deep

Here, I fix the number of LUT construction sub-PEs to be 16 to remove P&R error.

This should apply to M=32 SBERT as well, as the old PE fail for M=32 as well.  

(2) FIFO size in main function. 

I changed the first distance LUT FIFO from depth=8 to depth=512. Otherwise the performance drops as the LUT constructor cannot produce one row per cycle.

(3) Network input FIFO depth.

experiment: batch size = 2, nprobe = 128 QPS only 10 -> packet loss

384 * 4 * 128 = 192 KB per query

network input FIFO length=2048, 64 byte per unit -> 2048 * 64 / 1024 = 128 KB

For window size = 2: 192 * 2 / 128 = 3 -> need to extend the FIFO depth = 3 * 2048 = 6144

But to save resources, and since other part of the accelerator (network kernel, etc.) has other buffers, I set the FIFO length to 4096


# accelerator_flush_explicit

LUT construction + PQ Scan + K-selection. Here, PQ scan has the option of enabling double buffering (which typically cause P&R error with network). Also the results are fully sorted.

See entire_accelerator_final for the version without network.

## Register flush settings 

default setting: Free-Running/ Flushable Pipeline

There are several places connected to AXI, where only Flushable Pipeline is the option, so I explicitly set the style for them:

```
pipelineConfig.tcl:# config_compile -pipeline_style flp 
src/hls/DRAM_utils.hpp:#pragma HLS pipeline II=1 style=flp
src/hls/DRAM_utils.hpp:#pragma HLS pipeline II=1 style=flp
src/hls/DRAM_utils.hpp:#pragma HLS pipeline II=1 style=flp
src/hls/hierarchical_priority_queue.hpp:#pragma HLS pipeline II=1 style=flp
```