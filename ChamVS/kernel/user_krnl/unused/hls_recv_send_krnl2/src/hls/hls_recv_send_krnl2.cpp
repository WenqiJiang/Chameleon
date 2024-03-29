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

/*
Debug: 
     recv & send with different s_data

Receive data from connection A, then forward the data to connection B
Support 1 connection per direction only.
*/

void traffic_gen(int pkgWordCount, ap_uint<64> expectedTxPkgCnt, hls::stream<ap_uint<512> >& s_data_in)
{
#pragma HLS dataflow

     for (int i = 0; i < expectedTxPkgCnt; ++i)
     {
          for (int j = 0; j < pkgWordCount; ++j)
          {

               ap_uint<512> s_data;
               for (int k = 0; k < (512/32); k++)
               {
                    #pragma HLS UNROLL
                    s_data(k*32+31, k*32) = i*pkgWordCount+j;
               }
               s_data_in.write(s_data);
          }
     }
}


extern "C" {
void hls_recv_send_krnl2(
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
               // Rx
               int basePortRx, 
               ap_uint<64> expectedRxByteCnt, // for input & output
               // Tx
               int baseIpAddressTx,
               int basePortTx, 
               int pkgWordCountTx // number of 64-byte words per packet, e.g, 16 or 22
                      ) {


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
#pragma HLS INTERFACE s_axilite port=basePortRx bundle = control
#pragma HLS INTERFACE s_axilite port=expectedRxByteCnt bundle = control
#pragma HLS INTERFACE s_axilite port=baseIpAddressTx bundle=control
#pragma HLS INTERFACE s_axilite port=basePortTx bundle = control
#pragma HLS INTERFACE s_axilite port=pkgWordCountTx bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

static hls::stream<ap_uint<512> >    s_data;
#pragma HLS STREAM variable=s_data depth=512

#pragma HLS dataflow

          const int useConn = 1;

////////////////////     Recv     ////////////////////
          
          listenPorts(
               basePortRx, 
               useConn, 
               m_axis_tcp_listen_port, 
               s_axis_tcp_port_status);

          recvData(expectedRxByteCnt, 
               // s_data,
               s_axis_tcp_notification, 
               m_axis_tcp_read_pkg, 
               s_axis_tcp_rx_meta, 
               s_axis_tcp_rx_data);

////////////////////     Send     ////////////////////

          ap_uint<64> expectedTxByteCnt = expectedRxByteCnt;
          ap_uint<16> sessionID [8];

          openConnections(
               useConn, 
               baseIpAddressTx, 
               basePortTx, 
               m_axis_tcp_open_connection, 
               s_axis_tcp_open_status, 
               sessionID);

          ap_uint<64> expectedTxPkgCnt = expectedTxByteCnt / pkgWordCount / 64;
          traffic_gen( pkgWordCountTx, expectedTxPkgCnt, s_data);

          sendData(
               m_axis_tcp_tx_meta, 
               m_axis_tcp_tx_data, 
               s_axis_tcp_tx_status, 
               s_data, 
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