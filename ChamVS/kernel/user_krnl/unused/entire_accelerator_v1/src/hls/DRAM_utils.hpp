#pragma once

#include "constants.hpp"
#include "types.hpp"

void get_cell_addr_and_size(
    // in init
    int query_num, 
    int nlist,
    int nprobe,
    hls::stream<int> &s_nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_num_vecs,
    // in runtime
    hls::stream<int> &s_cell_ID,
    // out
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_last_valid_PE_ID,
    hls::stream<int> &s_start_addr_every_cell,
    hls::stream<int> &s_control_iter_num_per_query);

void load_PQ_codes(
    // in init
    int query_num, 
    int nprobe,
    // in runtime
    hls::stream<int> &s_cell_ID,
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_last_valid_PE_ID,
    hls::stream<int> &s_start_addr_every_cell,
    const ap_uint<512>* PQ_codes_DRAM_0,
    const ap_uint<512>* PQ_codes_DRAM_1,
    const ap_uint<512>* PQ_codes_DRAM_2,
    const ap_uint<512>* PQ_codes_DRAM_3,
    // out
    hls::stream<PQ_in_t> (&s_PQ_codes)[ADC_PE_NUM]);

void load_distance_LUT(
    int query_num, 
    int nprobe,
    ap_uint<512>* LUT_DRAM, // query_num * nprobe * 256 * M
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT);

void load_cell_ID(
    int query_num,
    int nprobe,
    int* cell_ID_DRAM,
    hls::stream<int>& s_cell_ID_get_cell_addr_and_size,
    hls::stream<int>& s_cell_ID_load_PQ_codes);

void load_nlist_init(
    int* nlist_init,
    hls::stream<int> &s_nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_vec_ID_start_addr,
    hls::stream<int> &s_nlist_num_vecs);


void load_nlist_PQ_codes_start_addr(
    int nlist,
    int* nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_PQ_codes_start_addr);

void load_nlist_num_vecs(
    int nlist,
    int* nlist_num_vecs,
    hls::stream<int> &s_nlist_num_vecs);

void load_nlist_vec_ID_start_addr(
    int nlist,
    int* nlist_vec_ID_start_addr,
    hls::stream<int> &s_nlist_vec_ID_start_addr);

void write_result(
    int query_num,
    hls::stream<result_t> &output_stream, 
    ap_uint<64>* out_DRAM);


void get_cell_addr_and_size(
    // in init
    int query_num, 
    int nlist,
    int nprobe,
    hls::stream<int> &s_nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_num_vecs,
    // in runtime
    hls::stream<int> &s_cell_ID,
    // out
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_last_valid_PE_ID,
    hls::stream<int> &s_start_addr_every_cell,
    hls::stream<int> &s_control_iter_num_per_query) {
    // Given the nprobe cell ID per query, output
    //    (1) startin addr in DRAM (entry ID for ap_uint<512>)
    //    (2) number of vectors to scan per cell

    // init a table that maps cell_ID to PQ_code start_address (entry ID for ap_uint<512>)
    int cell_ID_to_addr[NLIST_MAX];
#pragma HLS resource variable=cell_ID_to_addr core=RAM_2P_URAM
    int cell_ID_to_num_vecs[NLIST_MAX]; // number of vectors per cell
#pragma HLS resource variable=cell_ID_to_num_vecs core=RAM_2P_URAM
    for (int i = 0; i < nlist; i++) {
        cell_ID_to_addr[i] = s_nlist_PQ_codes_start_addr.read();
	    cell_ID_to_num_vecs[i] = s_nlist_num_vecs.read();
    }

    for (int query_id = 0; query_id < query_num; query_id++) {

        int iter_num_per_query = 0;
        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int cell_ID = s_cell_ID.read();
            int num_vec = cell_ID_to_num_vecs[cell_ID];

            int scanned_entries_every_cell;
            int last_valid_PE_ID; // from 0 to ADC_PE_NUM
            if (num_vec % ADC_PE_NUM == 0) {
                scanned_entries_every_cell = num_vec / ADC_PE_NUM;
                last_valid_PE_ID = ADC_PE_NUM - 1;
            }
            else {
                scanned_entries_every_cell = num_vec / ADC_PE_NUM + 1;
                last_valid_PE_ID = num_vec % ADC_PE_NUM - 1;
            }
            
            iter_num_per_query += scanned_entries_every_cell;

            s_scanned_entries_every_cell.write(scanned_entries_every_cell);
            s_last_valid_PE_ID.write(last_valid_PE_ID);

            int start_addr = cell_ID_to_addr[cell_ID];
            s_start_addr_every_cell.write(start_addr);
        }
	    s_control_iter_num_per_query.write(iter_num_per_query);
    }
}

void load_PQ_codes(
    // in init
    int query_num, 
    int nprobe,
    // in runtime
    hls::stream<int> &s_cell_ID,
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_last_valid_PE_ID,
    hls::stream<int> &s_start_addr_every_cell,
    const ap_uint<512>* PQ_codes_DRAM_0,
    const ap_uint<512>* PQ_codes_DRAM_1,
    const ap_uint<512>* PQ_codes_DRAM_2,
    const ap_uint<512>* PQ_codes_DRAM_3,
    // out
    hls::stream<PQ_in_t> (&s_PQ_codes)[ADC_PE_NUM]
) {


    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int cell_ID = s_cell_ID.read();
            int compute_iter_per_PE = s_scanned_entries_every_cell.read();
            int last_valid_PE_ID = s_last_valid_PE_ID.read();
            int start_addr = s_start_addr_every_cell.read();

            for (int entry_id = 0; entry_id < compute_iter_per_PE; entry_id++) {
#pragma HLS pipeline II=1
                ap_uint<512> PQ_reg_multi_PE_0 = PQ_codes_DRAM_0[start_addr + entry_id];
                ap_uint<512> PQ_reg_multi_PE_1 = PQ_codes_DRAM_1[start_addr + entry_id];
                ap_uint<512> PQ_reg_multi_PE_2 = PQ_codes_DRAM_2[start_addr + entry_id];
                ap_uint<512> PQ_reg_multi_PE_3 = PQ_codes_DRAM_3[start_addr + entry_id];

                for (int a = 0; a < ADC_PE_PER_CHANNEL; a++) {
#pragma HLS unroll
                    int s_ID_0 = a + 0 * ADC_PE_PER_CHANNEL;
                    int s_ID_1 = a + 1 * ADC_PE_PER_CHANNEL;
                    int s_ID_2 = a + 2 * ADC_PE_PER_CHANNEL;
                    int s_ID_3 = a + 3 * ADC_PE_PER_CHANNEL;

                    PQ_in_t PQ_reg_0;
                    PQ_in_t PQ_reg_1;
                    PQ_in_t PQ_reg_2;
                    PQ_in_t PQ_reg_3;
                    
                    PQ_reg_0.valid = ((entry_id == compute_iter_per_PE - 1) && (s_ID_0 > last_valid_PE_ID))? 0 : 1;
                    PQ_reg_1.valid = ((entry_id == compute_iter_per_PE - 1) && (s_ID_1 > last_valid_PE_ID))? 0 : 1;
                    PQ_reg_2.valid = ((entry_id == compute_iter_per_PE - 1) && (s_ID_2 > last_valid_PE_ID))? 0 : 1;
                    PQ_reg_3.valid = ((entry_id == compute_iter_per_PE - 1) && (s_ID_3 > last_valid_PE_ID))? 0 : 1;
                    
                    PQ_reg_0.cell_ID = cell_ID; 
                    PQ_reg_1.cell_ID = cell_ID; 
                    PQ_reg_2.cell_ID = cell_ID; 
                    PQ_reg_3.cell_ID = cell_ID; 

                    PQ_reg_0.offset = entry_id * ADC_PE_PER_CHANNEL + a; // per channel offset
                    PQ_reg_1.offset = entry_id * ADC_PE_PER_CHANNEL + a; // per channel offset
                    PQ_reg_2.offset = entry_id * ADC_PE_PER_CHANNEL + a; // per channel offset
                    PQ_reg_3.offset = entry_id * ADC_PE_PER_CHANNEL + a; // per channel offset

                    // bit refer: https://github.com/WenqiJiang/FPGA-ANNS/blob/main/integrated_accelerator/entire-node-systolic-computation-without-FIFO-type-assignment-fine-grained-PE-with-queue-group-inlined/src/HBM_interconnections.hpp
                    for (int m = 0; m < M; m++) {
#pragma HLS unroll
                        PQ_reg_0.PQ_code[m] = PQ_reg_multi_PE_0.range(
                            a * M * 8 + m * 8 + 7, a * M * 8 + m * 8);
                        PQ_reg_1.PQ_code[m] = PQ_reg_multi_PE_1.range(
                            a * M * 8 + m * 8 + 7, a * M * 8 + m * 8);
                        PQ_reg_2.PQ_code[m] = PQ_reg_multi_PE_2.range(
                            a * M * 8 + m * 8 + 7, a * M * 8 + m * 8);
                        PQ_reg_3.PQ_code[m] = PQ_reg_multi_PE_3.range(
                            a * M * 8 + m * 8 + 7, a * M * 8 + m * 8);
                    }
                    s_PQ_codes[s_ID_0].write(PQ_reg_0);
                    s_PQ_codes[s_ID_1].write(PQ_reg_1);
                    s_PQ_codes[s_ID_2].write(PQ_reg_2);
                    s_PQ_codes[s_ID_3].write(PQ_reg_3);
                }
            }
        }
    }
}


void load_distance_LUT(
    int query_num, 
    int nprobe,
    ap_uint<512>* LUT_DRAM, // query_num * nprobe * 256 * M
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT) {

    const int entries_per_row = 512 / 8 / sizeof(float) / M;

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

#if M == 8
            distance_LUT_parallel_t dist_row_A;
            distance_LUT_parallel_t dist_row_B;
            // one 512-bit entry = two PQ code row (8 floats x 4 byte = 32 bytes)
            for (int row_id = 0; row_id < LUT_ENTRY_NUM / 2; row_id++) {
#pragma HLS pipeline II=1
                int base_addr = (query_id * nprobe + nprobe_id) * (LUT_ENTRY_NUM / 2);
                ap_uint<512> reg = LUT_DRAM[base_addr + row_id];
                for (int n = 0; n < 8; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row_A.dist[n] = float_dist;
                }
                s_distance_LUT.write(dist_row_A);
                for (int n = 8; n < 16; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row_B.dist[n] = float_dist;
                }
                s_distance_LUT.write(dist_row_B);
            }
#elif M == 16
            distance_LUT_parallel_t dist_row;
            // one 512-bit entry = one PQ code row (16 floats x 4 byte = 64 bytes)
            for (int row_id = 0; row_id < LUT_ENTRY_NUM; row_id++) {
#pragma HLS pipeline II=1
                int base_addr = (query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM;
                ap_uint<512> reg = LUT_DRAM[base_addr + row_id];
                for (int n = 0; n < 16; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row.dist[n] = float_dist;
                }
                s_distance_LUT.write(dist_row);
            }
#elif M == 32
            distance_LUT_parallel_t dist_row;
            // two 512-bit entry = one PQ code row (32 floats x 4 byte = 128 bytes)
            for (int row_id = 0; row_id < LUT_ENTRY_NUM; row_id++) {
#pragma HLS pipeline II=1
                int base_addr = (query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM * 2;
                ap_uint<512> reg_A = LUT_DRAM[base_addr + row_id];
                ap_uint<512> reg_B = LUT_DRAM[base_addr + row_id + 1];
                for (int n = 0; n < 16; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg_A.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row.dist[n] = float_dist;
                }
                for (int n = 0; n < 16; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg_B.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row.dist[n + 16] = float_dist;
                }
                s_distance_LUT.write(dist_row);
            }
#elif M == 64
            distance_LUT_parallel_t dist_row;
            // four 512-bit entry = one PQ code row (32 floats x 4 byte = 128 bytes)
            for (int row_id = 0; row_id < LUT_ENTRY_NUM; row_id++) {
#pragma HLS pipeline II=1
                int base_addr = (query_id * nprobe + nprobe_id) * LUT_ENTRY_NUM * 4;
                ap_uint<512> reg_A = LUT_DRAM[base_addr + row_id];
                ap_uint<512> reg_B = LUT_DRAM[base_addr + row_id + 1];
                ap_uint<512> reg_C = LUT_DRAM[base_addr + row_id + 2];
                ap_uint<512> reg_D = LUT_DRAM[base_addr + row_id + 3];
                for (int n = 0; n < 16; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg_A.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row.dist[n] = float_dist;
                }
                for (int n = 0; n < 16; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg_B.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row.dist[n + 16] = float_dist;
                }
                for (int n = 0; n < 16; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg_C.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row.dist[n + 32] = float_dist;
                }
                for (int n = 0; n < 16; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg_D.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row.dist[n + 48] = float_dist;
                }
                s_distance_LUT.write(dist_row);
            }
#endif
        }
    }
}


void load_cell_ID(
    int query_num,
    int nprobe,
    int* cell_ID_DRAM,
    hls::stream<int>& s_cell_ID_get_cell_addr_and_size,
    hls::stream<int>& s_cell_ID_load_PQ_codes) {
        
    for (int query_id = 0; query_id < query_num; query_id++) {
        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {
            int cell_ID = cell_ID_DRAM[query_id * nprobe + nprobe_id];
            s_cell_ID_get_cell_addr_and_size.write(cell_ID); 
            s_cell_ID_load_PQ_codes.write(cell_ID); 
        }
    }
}

void load_nlist_PQ_codes_start_addr(
    int nlist,
    int* nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_PQ_codes_start_addr) {

    for (int i = 0; i < nlist; i++) {
#pragma HLS pipeline
        s_nlist_PQ_codes_start_addr.write(nlist_PQ_codes_start_addr[i]);
    }
}

void load_nlist_num_vecs(
    int nlist,
    int* nlist_num_vecs,
    hls::stream<int> &s_nlist_num_vecs) {

    for (int i = 0; i < nlist; i++) {
#pragma HLS pipeline
        s_nlist_num_vecs.write(nlist_num_vecs[i]);
    }
}


void load_nlist_vec_ID_start_addr(
    int nlist,
    int* nlist_vec_ID_start_addr,
    hls::stream<int> &s_nlist_vec_ID_start_addr) {

    for (int i = 0; i < nlist; i++) {
#pragma HLS pipeline
        s_nlist_vec_ID_start_addr.write(nlist_vec_ID_start_addr[i]);
    }
}

void load_nlist_init(
    int nlist,
    int* nlist_init,
    hls::stream<int> &s_nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_vec_ID_start_addr,
    hls::stream<int> &s_nlist_num_vecs) {

    // nlist_init = three of the following 
    // int* nlist_PQ_codes_start_addr,
    // int* nlist_vec_ID_start_addr,
    // int* nlist_num_vecs,

    int offset_nlist_PQ_codes_start_addr = 0;
    int offset_nlist_vec_ID_start_addr = nlist;
    int offset_nlist_num_vecs = 2 * nlist; 

    for (int i = 0; i < nlist; i++) {
#pragma HLS pipeline
        s_nlist_PQ_codes_start_addr.write(nlist_init[i + offset_nlist_PQ_codes_start_addr]);
        s_nlist_vec_ID_start_addr.write(nlist_init[i + offset_nlist_vec_ID_start_addr]);
        s_nlist_num_vecs.write(nlist_init[i + offset_nlist_num_vecs]);
    }

}

void write_result(
    int query_num,
    hls::stream<result_t> &output_stream, 
    ap_uint<64>* out_DRAM) {

    // only write the last iteration
    for (int i = 0; i < query_num * PRIORITY_QUEUE_LEN_L2; i++) {
#pragma HLS pipeline II=1
        result_t raw_output = output_stream.read();
        ap_uint<64> reg;
        int vec_ID = raw_output.vec_ID;
        float dist = raw_output.dist;
        reg.range(31, 0) = *((ap_uint<32>*) (&vec_ID));
        reg.range(63, 32) = *((ap_uint<32>*) (&dist));
        out_DRAM[i] = reg;
    }
}


