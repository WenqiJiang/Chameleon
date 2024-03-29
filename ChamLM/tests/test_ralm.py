import numpy as np
import torch

from ralm.ralm.ralm import ralmDecoder, ralmEncoderDecoder
from ralm.ralm.ralm_tiktok import ralmTikTokDecoder, ralmTikTokEncoderDecoder
from ralm.lm.get_model import createTransformerDecoder, createTransformerEncoderDecoder
from ralm.server.server import RandomAnswerServer
from ralm.retriever.retriever import ExternalRetriever, DummyRetriever
from ralm.index_scanner.index_scanner import IndexScanner
from threading import Thread
import time
import argparse 


def test_local_ralm():
    """
    Test decoder-only / encoder-decoder models, using a dummy retriever.
    """
    batch_size = 32
    dim = 512
    layers = 12
    attention_heads = 16
    k = 10
    retrieval_interval = 1
    retrieval_token_len = 16
    seq_len = 512
    nlist = 32768
    nprobe = 32
    n_batch = 1
    use_gpu_id = 0
    
    gpu_index_scanner = IndexScanner(dim=dim, nlist=nlist, nprobe=nprobe, use_gpu_id=use_gpu_id)
    if torch.cuda.is_available():
        if use_gpu_id is not None:
            device = f'cuda:{use_gpu_id}'
        else:
            device = 'cuda'
    else:
        device = 'cpu'

    for model_type in ["decoder", "encoder-decoder"]:

        for request_with_lists in [0, 1]:
                        
            parser = argparse.ArgumentParser()
            args_model = parser.parse_args()
            args_model.decoder = { \
                "embed_dim" : dim, 
                "ffn_embed_dim": dim * 4, 
                "layers" : layers,
                "attention_heads" : attention_heads}
            if model_type == "encoder-decoder":
                args_model.encoder = { \
                    "embed_dim" : dim, 
                    "ffn_embed_dim": dim * 4, 
                    "layers" : layers,
                    "attention_heads" : attention_heads}
                

            print(f"Executing on {device}")

            retriever = DummyRetriever(default_k=k)
            index_scanner = gpu_index_scanner if request_with_lists else None

            if model_type == "encoder-decoder":
                print("Creating encoder-decoder model...")
                model_encoder, model_decoder = createTransformerEncoderDecoder(args=args_model)
                ralm = ralmEncoderDecoder(model_encoder, model_decoder, retriever, index_scanner=index_scanner,
                    batch_size=batch_size, retrieval_interval=retrieval_interval, retrieval_token_len=retrieval_token_len,
                    device=device)
            else:
                print("Creating decoder model...")
                model = createTransformerDecoder(args=args_model)
                ralm = ralmDecoder(model, retriever, index_scanner=index_scanner, batch_size=batch_size, retrieval_interval=retrieval_interval, device=device)

            # functionality test
            ralm.single_step()
            ralm.multi_steps(num_step=seq_len - 1)
            ralm.reset_inference_state()

            # batch processing
            print("Start batch inference...")
            for i in range(n_batch):
                print(f"Batch {i}")
                ralm.batch_inference(num_step=seq_len)


# def test_tiktok_ralm():
#     """
#     Test TikTok RALM model, by combining them with external retrievers.
#     """

#     """
#     Starts the server
#     """    

#     """
#     The RALM Model
#     """
    
#     host = "127.0.0.1"
#     port = 9099
#     batch_size = 32
#     dim = 512
#     layers = 12
#     attention_heads = 16
#     k = 100
#     retrieval_interval = 2
#     retrieval_token_len = 32
#     seq_len = 256
#     n_batch = 1
#     use_gpu_id = 0
#     use_tiktok = 1
#     delay_ms = 0

#     nlist = 32768
#     nprobe = 32
    
#     gpu_index_scanner = IndexScanner(dim=dim, nlist=nlist, nprobe=nprobe, use_gpu_id=use_gpu_id)
    

#     n_retrieve_per_seq = seq_len / retrieval_interval
#     print("The server needs to answer {} batches of queries".format(int(n_retrieve_per_seq * n_batch)))
#     print("Please start the server before this process..., e.g., \n"
#         "cd ralm/server\n"
#         "python server.py")

#     # for model_type in ["decoder", "encoder-decoder"]:
#     for model_type in ["decoder"]:

#         for request_with_lists in [0]:
#         # for request_with_lists in [0, 1]:

#             port += 1
#             random_answer_server = RandomAnswerServer(
#                 host=host, port=port, batch_size=batch_size, dim=dim, delay_ms=delay_ms)
#             # random_answer_server.start_one_query_per_conn() # one query per connection
#             mock_server_thread = Thread(target=random_answer_server.start)
#             mock_server_thread.daemon = True
#             mock_server_thread.start()

#             retriever = ExternalRetriever(default_k=k, host=host, port=port, batch_size=batch_size, dim=dim)
#             index_scanner = gpu_index_scanner if request_with_lists else None

#             parser = argparse.ArgumentParser()
#             args_model = parser.parse_args()
#             args_model.decoder = { \
#                 "embed_dim" : dim, 
#                 "ffn_embed_dim": dim * 4, 
#                 "layers" : layers,
#                 "attention_heads" : attention_heads}
#             if model_type == "encoder-decoder":
#                 args_model.encoder = { \
#                     "embed_dim" : dim, 
#                     "ffn_embed_dim": dim * 4, 
#                     "layers" : layers,
#                     "attention_heads" : attention_heads}
                
#             if torch.cuda.is_available():
#                 if use_gpu_id is not None:
#                     device = f'cuda:{use_gpu_id}'
#                 else:
#                     device = 'cuda'
#             else:
#                 device = 'cpu'

#             print(f"Executing on {device}")

#             if use_tiktok:
#                 print("Using tiktok scheduling...")
#             else:
#                 print("Disabled tiktok scheduling...")

#             if model_type == "encoder-decoder":
#                 print("Creating encoder-decoder model...")
#                 model_encoder, model_decoder = createTransformerEncoderDecoder(args=args_model)
#                 if use_tiktok:
#                     ralm = ralmTikTokEncoderDecoder(model_encoder, model_decoder, retriever, index_scanner=index_scanner,
#                         batch_size=batch_size, retrieval_interval=retrieval_interval, retrieval_token_len=retrieval_token_len, 
#                         device=device)
#                 else:
#                     ralm = ralmEncoderDecoder(model_encoder, model_decoder, retriever, index_scanner=index_scanner,
#                         batch_size=batch_size, retrieval_interval=retrieval_interval, retrieval_token_len=retrieval_token_len, 
#                         device=device)
#             else:
#                 print("Creating decoder model...")
#                 model_decoder = createTransformerDecoder(args=args_model)
#                 if use_tiktok:
#                     ralm = ralmTikTokDecoder(model_decoder, retriever, index_scanner=index_scanner, batch_size=batch_size, retrieval_interval=retrieval_interval, device=device)
#                 else:
#                     ralm = ralmDecoder(model_decoder, retriever, index_scanner=index_scanner, batch_size=batch_size, retrieval_interval=retrieval_interval, device=device)
    
    
#             ralm_thread = Thread(target=ralm.batch_inference, args=(seq_len, ))
#             ralm_thread.daemon = True
#             ralm_thread.start()
#             # TODO: problem with join threads...
#             ralm_thread.join()