/*
 * Copyright (c) 2020, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "ap_axi_sdata.h"
#include <ap_fixed.h>
#include "ap_int.h" 
#include "../../../../common/include/communication.hpp"
#include "hls_stream.h"

#include "ADC.hpp"
#include "constants.hpp"
#include "DRAM_utils.hpp"
#include "helpers.hpp"
#include "hierarchical_priority_queue.hpp"
#include "types.hpp"

/*
Receive data from connection A, then forward the data to connection B
Support 1 connection per direction only.
*/


void network_input_processing(
    int query_num,
    int nprobe,
    // in runtime
    hls::stream<ap_uint<512>>& s_kernel_network_in,
    // out
    hls::stream<int>& s_cell_ID,
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT
    ) {

    // Format: foe each query
    // packet 0: header (query_id, nprobe), in the future, nprobe is streamed from network
    // packet 1~k: cell_IDs to scan -> size = ceil(nprobe * 4 / 64) 
    // packet k~n: nprobe x LUTs -> size = nprobe * 4 * M * LUT_ENTRY_NUM / 64

    // size in 512-bit/64-byte chunks

    for (int query_id = 0; query_id < query_num; query_id++) {

        // header meta
        ap_uint<512> header = s_kernel_network_in.read();
        ap_uint<32> query_id_unused_uint = header.range(31, 0);
        ap_uint<32> nprobe_unused_uint = header.range(63, 32);
        int query_id_unused = *((int*) (&query_id_unused_uint));
        int nprobe_unused = *((int*) (&nprobe_unused_uint));

        int size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;
        int size_LUTs = nprobe * 4 * M * LUT_ENTRY_NUM / 64; // should always be int

        // cell_IDs
        for (int i = 0; i < size_cell_IDs; i++) {

            ap_uint<512> pkt = s_kernel_network_in.read();

            for (int j = 0; j < 16; j++) {

                ap_uint<32> cell_ID_uint = pkt.range(32 * j + 31, 32 * j);
                int cell_ID = *((int*) (&cell_ID_uint));

                int cell_count = i * 16 + j;
                if (cell_count < nprobe) {
                    s_cell_ID.write(cell_ID);
                }
            }
        }

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            // LUTs
                
#if M == 8
            for (int i = 0; i < LUT_ENTRY_NUM / 2; i++) {
                distance_LUT_parallel_t dist_row_A;
                distance_LUT_parallel_t dist_row_B;
                // one 512-bit entry = two PQ code row (8 floats x 4 byte = 32 bytes)
                ap_uint<512> reg = s_kernel_network_in.read();
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
                    dist_row_B.dist[n - 8] = float_dist;
                }
                s_distance_LUT.write(dist_row_B);
            } 
#elif M == 16
            for (int i = 0; i < LUT_ENTRY_NUM; i++) {
                distance_LUT_parallel_t dist_row;
                ap_uint<512> reg = s_kernel_network_in.read();
                for (int n = 0; n < 16; n++) {
#pragma HLS UNROLL
                    ap_uint<32> uint_dist = reg.range(32 * (n + 1) - 1, 32 * n);
                    float float_dist = *((float*) (&uint_dist));
                    dist_row.dist[n] = float_dist;
                }
                s_distance_LUT.write(dist_row);
            }
#elif M == 32
            for (int i = 0; i < LUT_ENTRY_NUM; i++) {
                distance_LUT_parallel_t dist_row;
                ap_uint<512> reg_A = s_kernel_network_in.read();
                ap_uint<512> reg_B = s_kernel_network_in.read();
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
            for (int i = 0; i < LUT_ENTRY_NUM; i++) {
                distance_LUT_parallel_t dist_row;
                ap_uint<512> reg_A = s_kernel_network_in.read();
                ap_uint<512> reg_B = s_kernel_network_in.read();
                ap_uint<512> reg_C = s_kernel_network_in.read();
                ap_uint<512> reg_D = s_kernel_network_in.read();
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

void network_output_processing(
    int query_num,
    hls::stream<result_t> &s_output, 
    hls::stream<ap_uint<512>>& s_kernel_network_out) {

    // Format: foe each query
    // packet 0: header (query_id, topK)
    // packet 1~k: topK results_pair (vec_ID, dist) -> size = ceil(topK * 8 / 64) 

    // in 512-byte packets
    const int size_results_vec_ID = PRIORITY_QUEUE_LEN_L2 * 64 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 64 / 512 : PRIORITY_QUEUE_LEN_L2 * 64 / 512 + 1;
    const int size_results_dist = PRIORITY_QUEUE_LEN_L2 * 32 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 32 / 512 : PRIORITY_QUEUE_LEN_L2 * 32 / 512 + 1;

    ap_uint<64> vec_ID_buffer [size_results_vec_ID * (512 / 64)] = { 0 };
    float dist_buffer[size_results_dist * (512 / 32)] = { 0 };

    // only write the last iteration
    for (int i = 0; i < query_num; i++) {
#pragma HLS pipeline II=1

        ap_uint<512> header = 0;
        ap_uint<32> query_id_header = i;
        ap_uint<32> topK_header = PRIORITY_QUEUE_LEN_L2;
        header.range(31, 0) = query_id_header;
        header.range(63, 32) = topK_header;
        s_kernel_network_out.write(header);

        for (int k = 0; k < TOPK; k++) {
            result_t raw_output = s_output.read();
            vec_ID_buffer[k] = raw_output.vec_ID;
            dist_buffer[k] = raw_output.dist;
        }

        // send vec IDs first
        for (int j = 0; j < size_results_vec_ID; j++) {
            ap_uint<512> pkt = 0;

            for (int k = 0; k < 512 / 64; k++) {
                
                    ap_uint<64> vec_ID = vec_ID_buffer[j * 512 / 64 + k];
                    pkt.range(64 * k + 63, 64 * k) = vec_ID;
                    
            }
            s_kernel_network_out.write(pkt);
        }

        // then send disk
        for (int j = 0; j < size_results_dist; j++) {
            ap_uint<512> pkt = 0;

            for (int k = 0; k < 512 / 32; k++) {
                
                    float dist = dist_buffer[j * 512 / 32 + k];
                    ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist));
                    pkt.range(32 * k + 31, 32 * k) = dist_uint;
            }
            s_kernel_network_out.write(pkt);
        }
    } 
}


extern "C" {
void entire_accelerator_v1(
     // Internal Stream
     hls::stream<pkt512>& s_axis_udp_rx, 
     hls::stream<pkt512>& m_axis_udp_tx, 
     hls::stream<pkt256>& s_axis_udp_rx_meta, 
     hls::stream<pkt256>& m_axis_udp_tx_meta, 
     
     hls::stream<pkt16>& m_axis_tcp_listen_port, 
     hls::stream<pkt8>& s_axis_tcp_port_status, 
     hls::stream<pkt64>& m_axis_tcp_open_connection, 
     hls::stream<pkt32>& s_axis_tcp_open_status, 
     hls::stream<pkt16>& m_axis_tcp_close_connection, 
     hls::stream<pkt128>& s_axis_tcp_notification, 
     hls::stream<pkt32>& m_axis_tcp_read_pkg, 
     hls::stream<pkt16>& s_axis_tcp_rx_meta, 
     hls::stream<pkt512>& s_axis_tcp_rx_data, 
     hls::stream<pkt32>& m_axis_tcp_tx_meta, 
     hls::stream<pkt512>& m_axis_tcp_tx_data, 
     hls::stream<pkt64>& s_axis_tcp_tx_status,
     // Rx & Tx
     int useConn,
     // Rx
     int basePortRx, 
     ap_uint<64> expectedRxByteCnt, // for input & output
     // Tx
     int baseIpAddressTx,
     int basePortTx, 
     ap_uint<64> expectedTxPkgCnt,
     int pkgWordCountTx, // number of 64-byte words per packet, e.g, 16 or 22

     //////////     Accelerator kernel     //////////
    // in init
    int query_num, 
    int nlist,
    int nprobe,
    int* nlist_init, // which consists of the following three stuffs:
                    // int* nlist_PQ_codes_start_addr,
                    // int* nlist_vec_ID_start_addr,
                    // int* nlist_num_vecs,

    // in runtime (should from DRAM)
    const ap_uint<512>* PQ_codes_DRAM_0,
    const ap_uint<512>* PQ_codes_DRAM_1,
    const ap_uint<512>* PQ_codes_DRAM_2,
    const ap_uint<512>* PQ_codes_DRAM_3,
    ap_uint<64>* vec_ID_DRAM_0,
    ap_uint<64>* vec_ID_DRAM_1,
    ap_uint<64>* vec_ID_DRAM_2,
    ap_uint<64>* vec_ID_DRAM_3
                      ) {

// network 
#pragma HLS INTERFACE axis port = s_axis_udp_rx
#pragma HLS INTERFACE axis port = m_axis_udp_tx
#pragma HLS INTERFACE axis port = s_axis_udp_rx_meta
#pragma HLS INTERFACE axis port = m_axis_udp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_listen_port
#pragma HLS INTERFACE axis port = s_axis_tcp_port_status
#pragma HLS INTERFACE axis port = m_axis_tcp_open_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_open_status
#pragma HLS INTERFACE axis port = m_axis_tcp_close_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_notification
#pragma HLS INTERFACE axis port = m_axis_tcp_read_pkg
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_meta
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_data
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_data
#pragma HLS INTERFACE axis port = s_axis_tcp_tx_status

// #pragma HLS INTERFACE s_axilite port=useConn bundle = control
// #pragma HLS INTERFACE s_axilite port=basePortRx bundle = control
// #pragma HLS INTERFACE s_axilite port=expectedRxByteCnt bundle = control
// #pragma HLS INTERFACE s_axilite port=baseIpAddressTx bundle=control
// #pragma HLS INTERFACE s_axilite port=basePortTx bundle = control
// #pragma HLS INTERFACE s_axilite port=expectedTxPkgCnt bundle = control
// #pragma HLS INTERFACE s_axilite port=pkgWordCountTx bundle = control
// #pragma HLS INTERFACE s_axilite port = return bundle = control

// accelerator kernel 

// #pragma HLS INTERFACE s_axilite port=query_num bundle = control
// #pragma HLS INTERFACE s_axilite port=nlist bundle = control
// #pragma HLS INTERFACE s_axilite port=nprobe bundle = control

#pragma HLS INTERFACE m_axi port=nlist_init offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_0 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_1 offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_2 offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_3 offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_0  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_1  offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_2  offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_3  offset=slave bundle=gmem12


#pragma HLS dataflow

////////////////////     Recv     ////////////////////
          
          listenPorts(
               basePortRx, 
               useConn, 
               m_axis_tcp_listen_port, 
               s_axis_tcp_port_status);

     hls::stream<ap_uint<512>> s_kernel_network_in;
#pragma HLS STREAM variable=s_kernel_network_in depth=512

          recvData(expectedRxByteCnt, 
               s_kernel_network_in,
               s_axis_tcp_notification, 
               m_axis_tcp_read_pkg, 
               s_axis_tcp_rx_meta, 
               s_axis_tcp_rx_data);


////////////////////     Network Input     ////////////////////

    hls::stream<int> s_cell_ID;
#pragma HLS stream variable=s_cell_ID depth=256
    
    hls::stream<distance_LUT_parallel_t> s_distance_LUT[ADC_PE_NUM + 1];
#pragma HLS stream variable=s_distance_LUT depth=8
#pragma HLS array_partition variable=s_distance_LUT complete
// #pragma HLS resource variable=s_distance_LUT core=FIFO_SRL

    // systolic array of distance LUT communication
    // load_distance_LUT(
    //     query_num, 
    //     nprobe,
    //     LUT_DRAM, // query_num * nprobe * 256 * M
    //     s_distance_LUT[0]);

    network_input_processing(
        query_num,
        nprobe,
        // in runtime
        s_kernel_network_in,
        // out
        s_cell_ID,
        s_distance_LUT[0]);

////////////////////     First Half: ADC     ////////////////////

    hls::stream<int> s_nlist_PQ_codes_start_addr;
#pragma HLS stream variable=s_nlist_PQ_codes_start_addr depth=256

    hls::stream<int> s_nlist_vec_ID_start_addr; // the top 10 numbers
#pragma HLS stream variable=s_nlist_vec_ID_start_addr depth=256
    
    hls::stream<int> s_nlist_num_vecs;
#pragma HLS stream variable=s_nlist_num_vecs depth=256

    load_nlist_init(
        nlist,
        nlist_init,
        s_nlist_PQ_codes_start_addr,
        s_nlist_vec_ID_start_addr,
        s_nlist_num_vecs);

    hls::stream<int> s_cell_ID_get_cell_addr_and_size;
#pragma HLS stream variable=s_cell_ID_get_cell_addr_and_size depth=256
    
    hls::stream<int> s_cell_ID_load_PQ_codes;
#pragma HLS stream variable=s_cell_ID_load_PQ_codes depth=256

    replicate_s_cell_ID(
        query_num,
        nprobe,
        s_cell_ID,
        s_cell_ID_get_cell_addr_and_size,
        s_cell_ID_load_PQ_codes);

    // load_cell_ID(
    //     query_num,
    //     nprobe,
    //     cell_ID_DRAM,
    //     s_cell_ID_get_cell_addr_and_size,
    //     s_cell_ID_load_PQ_codes);

    hls::stream<int> s_scanned_entries_every_cell;
#pragma HLS stream variable=s_scanned_entries_every_cell depth=256
// #pragma HLS resource variable=s_scanned_entries_every_cell core=FIFO_SRL
    
    hls::stream<int> s_last_valid_PE_ID;
#pragma HLS stream variable=s_last_valid_PE_ID depth=256
// #pragma HLS resource variable=s_last_valid_PE_ID core=FIFO_SRL
    
    hls::stream<int> s_start_addr_every_cell;
#pragma HLS stream variable=s_start_addr_every_cell depth=256
// #pragma HLS resource variable=s_start_addr_every_cell core=FIFO_SRL
    
    hls::stream<int> s_control_iter_num_per_query;
#pragma HLS stream variable=s_control_iter_num_per_query depth=256
// #pragma HLS resource variable=s_control_iter_num_per_query core=FIFO_SRL
    
    get_cell_addr_and_size(
        // in init
        query_num, 
	    nlist,
        nprobe,
        s_nlist_PQ_codes_start_addr,
        s_nlist_num_vecs,
        // in runtime
        s_cell_ID_get_cell_addr_and_size,
        // out
        s_scanned_entries_every_cell,
        s_last_valid_PE_ID,
        s_start_addr_every_cell,
        s_control_iter_num_per_query);

    hls::stream<int> s_scanned_entries_every_cell_ADC[ADC_PE_NUM];
#pragma HLS stream variable=s_scanned_entries_every_cell_ADC depth=256
#pragma HLS array_partition variable=s_scanned_entries_every_cell_ADC complete
// #pragma HLS resource variable=s_scanned_entries_every_cell_ADC core=FIFO_SRL

    hls::stream<int> s_scanned_entries_every_cell_load_PQ_codes;
#pragma HLS stream variable=s_scanned_entries_every_cell_load_PQ_codes depth=256
// #pragma HLS resource variable=s_scanned_entries_every_cell_load_PQ_codes core=FIFO_SRL

    replicate_s_scanned_entries_every_cell(
        // in
        query_num,
        nprobe,
        s_scanned_entries_every_cell,
        // out
        s_scanned_entries_every_cell_ADC,
        s_scanned_entries_every_cell_load_PQ_codes);

    hls::stream<PQ_in_t> s_PQ_codes[ADC_PE_NUM];
#pragma HLS stream variable=s_PQ_codes depth=8
#pragma HLS array_partition variable=s_PQ_codes complete
// #pragma HLS resource variable=s_PQ_codes core=FIFO_SRL

    load_PQ_codes(
        // in init
        query_num, 
        nprobe,
        // in runtime
        s_cell_ID_load_PQ_codes,
        s_scanned_entries_every_cell_load_PQ_codes,
        s_last_valid_PE_ID,
        s_start_addr_every_cell,
        PQ_codes_DRAM_0,
        PQ_codes_DRAM_1,
        PQ_codes_DRAM_2,
        PQ_codes_DRAM_3,
        // out
        s_PQ_codes);

    hls::stream<PQ_out_t> s_PQ_result[ADC_PE_NUM];
#pragma HLS stream variable=s_PQ_result depth=8
#pragma HLS array_partition variable=s_PQ_result complete
// #pragma HLS resource variable=s_PQ_result core=FIFO_SRL


    for (int s = 0; s < ADC_PE_NUM; s++) {
#pragma HLS unroll
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT[s],
            s_PQ_codes[s],
            s_scanned_entries_every_cell_ADC[s],
            // output streams
            s_distance_LUT[s + 1],
            s_PQ_result[s]);
    }

    dummy_distance_LUT_consumer(
        query_num, 
        nprobe,
        s_distance_LUT[ADC_PE_NUM]);

////////////////////     Second Half: K-Selection     ////////////////////

    hls::stream<result_t> s_output; // the topK numbers
#pragma HLS stream variable=s_output depth=256

    hierarchical_priority_queue( 
        query_num, 
        nlist,
        s_nlist_vec_ID_start_addr,
        s_control_iter_num_per_query, 
        s_PQ_result,
        vec_ID_DRAM_0,
        vec_ID_DRAM_1,
        vec_ID_DRAM_2,
        vec_ID_DRAM_3,
        s_output);

////////////////////     Network Output     ////////////////////

    hls::stream<ap_uint<512>> s_kernel_network_out; 
#pragma HLS stream variable=s_kernel_network_out depth=512

    network_output_processing(
        query_num,
        s_output, 
        s_kernel_network_out);

////////////////////     Send     ////////////////////

          ap_uint<16> sessionID [8];

          openConnections(
               useConn, 
               baseIpAddressTx, 
               basePortTx, 
               m_axis_tcp_open_connection, 
               s_axis_tcp_open_status, 
               sessionID);

          ap_uint<64> expectedTxByteCnt = expectedTxPkgCnt * pkgWordCountTx * 64;
          sendData(
               m_axis_tcp_tx_meta, 
               m_axis_tcp_tx_data, 
               s_axis_tcp_tx_status, 
               s_kernel_network_out, 
               sessionID,
               useConn, 
               expectedTxByteCnt, 
               pkgWordCountTx);


////////////////////     Tie off     ////////////////////

          tie_off_udp(s_axis_udp_rx, 
               m_axis_udp_tx, 
               s_axis_udp_rx_meta, 
               m_axis_udp_tx_meta);
    
          tie_off_tcp_close_con(m_axis_tcp_close_connection);

     }
}