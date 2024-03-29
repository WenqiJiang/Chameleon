"""
Reproduce the forward pass of RALM, both decoder-only and encoder-decoder. 
"""

import torch 
import time
import numpy as np

from typing import Optional


@torch.no_grad()
class ralmDecoder:
    """
    A decoder-only RALM, support arbitrary decoder model, retrieval module, and retrieval frequency.
    """
    
    @torch.no_grad()
    def __init__(self, model, retriever, index_scanner = None, query_set: Optional[np.array] = None, batch_size : Optional[int] = 1, 
                 retrieval_interval : Optional[int] = 1, device : Optional[str] = 'cpu', use_coordinator : Optional[int] = 0):
        """
        Input arguments:
            model : a TransformerDecoder
            Retriever : a retriever
            index_scanner: a Faiss index scanner (can only be used by coupling with an external/dummy retriever)
            batch_size : inference batch size 
            retrieval_interval : retrieve every N steps
            use_coordinator : whether to use a coordinator to manage the retrieval (which needs extra synchronization)
        """
        self.model = model
        self.retriever = retriever
        self.device = device

        self.index_scanner = index_scanner
        self.request_with_lists = True if index_scanner is not None else False

        self.query_set = query_set
        
        self.batch_size = batch_size
        self.retrieval_interval = retrieval_interval

        self.dim = model.args.decoder['embed_dim']
        self.model.to(self.device)

        # Set up model inference & retrieval states
        self.reset_inference_state()

        # warmup
        self.model(self.input_tokens, incremental_state=self.incremental_state)
        
        # Last step: if interacting with cooridnator, wait until the coordinator and all GPUs are ready
        if use_coordinator:
            self.retriever.sync_with_coordinator()
        
    @torch.no_grad()
    def reset_inference_state(self):
        """
        Reset all the inference states: start inference from the first token
        The state must be reset if the batch size / retrieval interval is changed

        Run this as an initialization and after each batch inference
        """
        # Set up model inference 
        start_seq_len = 1
        begin_of_sentence = 0
        self.input_tokens = torch.tensor([[begin_of_sentence] * start_seq_len] * self.batch_size).to(self.device)
        self.incremental_state = dict()

        # three profile arrays
        self.time_model = [] # model inference per step in sec 
        self.time_retriever = [] # retrieval per step (0 if no retrieval) in sec 
        self.time_step = [] # total time per step

        # Retrieval parameters
        self.current_step = 0

        if self.query_set is not None:
            nq, d = self.query_set.shape
            assert d == self.dim
            assert nq > self.batch_size
            self.query_set_start_id = 0
            self.query_set_end_id = self.batch_size

    @torch.no_grad()
    def single_step(self):
        """
        Combine forward pass and (optionally) a retrieval, for single token generation.
        """
        
        """
        The incremental_state will only add the current state into the dictionary, 
            i.e., its size grows linearly with its step steps. 
        Thus, it incremental inference must start from the beginning, i.e.,
            not with a prefix of 500

        Example:
            step 510: 32.01556205749512 ms
            Shape of one layer in incremental_state:  torch.Size([1, 16, 510, 64])
            step 511: 31.69989585876465 ms
            Shape of one layer in incremental_state:  torch.Size([1, 16, 511, 64])
            step 512: 30.326128005981445 ms
            Shape of one layer in incremental_state:  torch.Size([1, 16, 512, 64])
        """
        start_step = time.time()

        start_model = time.time()
        out_tensor, out_dict = self.model(self.input_tokens, incremental_state=self.incremental_state)
        # get output state of the model
        if self.query_set is None:
            state_for_retrieval = out_dict['inner_states'][-1] # [1, Batch_size, Hidden_dim]
            state_for_retrieval = state_for_retrieval.view(self.batch_size, self.dim).cpu().numpy()
        else:
            state_for_retrieval = self.query_set[self.query_set_start_id:self.query_set_end_id]
            self.query_set_start_id += self.batch_size
            self.query_set_end_id += self.batch_size
            if self.query_set_end_id > self.query_set.shape[0]:
                self.query_set_start_id = 0
                self.query_set_end_id = self.batch_size
        end_model = time.time()

        if self.current_step % self.retrieval_interval == 0:
            start_retriever = time.time()
            if self.request_with_lists:
                list_IDs, list_centroids = self.index_scanner.search(state_for_retrieval)
                self.retrieval_cache = self.retriever.retrieve_with_lists(state_for_retrieval, list_IDs)
            else:
                self.retrieval_cache = self.retriever.retrieve(state_for_retrieval)
            end_retriever = time.time()
            self.time_retriever.append(end_retriever - start_retriever)
        else:
            self.time_retriever.append(0)

        # TODO: optionally implement a combiner

        # expand the input token sequence
        self.current_step += 1
        next_dummy_token = 1 # always set the next token ID as 1
        self.input_tokens = torch.tensor([[next_dummy_token] * 1] * self.batch_size).to(self.device)

        end_step = time.time()

        self.time_model.append(end_model - start_model)
        self.time_step.append(end_step - start_step)

    @torch.no_grad()
    def multi_steps(self, num_step : Optional[int] = 1, nprobe : Optional[int] = 1):
        """
        Combine forward pass and (optionally) a retrieval, for multiple token generations.
        """
        for i in range(num_step):
            self.single_step()
        
    @torch.no_grad()
    def batch_inference(self, num_step : Optional[int] = 1, nprobe : Optional[int] = 1):
        """
        multi_steps with a reset_inference_state
        """
        self.reset_inference_state()
        self.time_start_batch = time.time()
        self.multi_steps(num_step, nprobe)
        self.time_end_batch = time.time()

    def get_profiling(self):
        """
        Return the following three arrays:
            model inference per step in sec 
            retrieval per step (0 if no retrieval) in sec 
            total time per step
        """
        return self.time_model, self.time_retriever, self.time_step
    
    def print_profiling_stats(self):
        """
        Print some stats of profiling
        """
        assert len(self.time_model) == len(self.time_retriever)
        assert len(self.time_model) == len(self.time_step)

        # in ms
        time_model = np.array(self.time_model) * 1000
        time_retriever = np.array(self.time_retriever) * 1000
        time_step = np.array(self.time_step) * 1000
        time_other = time_step - time_model - time_retriever
        
        time_batch = (self.time_end_batch - self.time_start_batch) * 1000
        print("== Batch profiling: ==")
        print("Batch size: {}".format(self.batch_size))
        print("Throughput (steps * batch_size / sec): {:.2f}".format(
            self.batch_size * len(time_step) / (time_batch / 1000)))
        print("Latency (ms/step): {:.2f}".format(time_batch/len(time_step)))
        print("Total time steps: {}".format(len(time_model)))
        print("Entire batch time: {:.2f} ms".format(time_batch))
        print("\n")

        print("Step (overall): total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_step), np.average(time_step)))
        print("Inference: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_model), np.average(time_model)))
        print("Retrieval: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_retriever), np.average(time_retriever)))
        print("Other: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_other), np.average(time_other)))



@torch.no_grad()
class ralmEncoderDecoder:
    """
    A encoder-decoder RALM, support arbitrary encoder model, decoder model, retrieval module, and retrieval frequency.
    """
    
    @torch.no_grad()
    def __init__(self, model_encoder, model_decoder, retriever, index_scanner = None, query_set: Optional[np.array] = None, batch_size : Optional[int] = 1, 
                 retrieval_interval : Optional[int] = 1, retrieval_token_len : Optional[int] = 64, 
                 k : Optional[int] = 10, device : Optional[str] = 'cpu', use_coordinator : Optional[int] = 0):
        """
        Input arguments:
            model : a TransformerDecoder
            Retriever : a retriever
            batch_size : inference batch size 
            retrieval_interval : retrieve every N steps
            retrieval_token_len: the token length per retrieved texts
        """
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.retriever = retriever
        self.device = device

        self.index_scanner = index_scanner
        self.request_with_lists = True if index_scanner is not None else False
       
        self.query_set = query_set 

        self.batch_size = batch_size
        self.retrieval_token_len = retrieval_token_len
        self.retrieval_interval = retrieval_interval
        self.k = k

        assert model_decoder.args.decoder['embed_dim'] == model_encoder.args.encoder['embed_dim']
        self.dim = model_decoder.args.decoder['embed_dim']

        self.model_encoder.to(self.device)
        self.model_decoder.to(self.device)

        # Set up model inference & retrieval states
        self.reset_inference_state()

        # Last step: if interacting with cooridnator, wait until the coordinator and all GPUs are ready
        if use_coordinator:
            self.retriever.sync_with_coordinator()


    @torch.no_grad()
    def reset_inference_state(self):
        """
        Reset all the inference states: start inference from the first token
        The state must be reset if the batch size / retrieval interval is changed
        """
        # Set up model inference 
        start_seq_len = 1
        begin_of_sentence = 0
        dummy_token = 1
        self.input_tokens = torch.tensor([[begin_of_sentence] * start_seq_len] * self.batch_size).to(self.device)
        # query tokens have identical length as each retrieved chunk, but there are k retrieved chunks
        self.query_tokens = torch.tensor([[dummy_token] * self.retrieval_token_len] * self.batch_size).to(self.device)
        self.retrieved_tokens = torch.tensor([[dummy_token] * self.retrieval_token_len * self.k] * self.batch_size).to(self.device)
        self.incremental_state = dict()

        # profile arrays
        self.time_model = [] # model inference per step in sec (encoder + decoder)
        self.time_model_encoder = [] # model inference per step in sec (encoder)
        self.time_model_decoder = [] # model inference per step in sec (decoder)
        self.time_retriever = [] # retrieval per step (0 if no retrieval) in sec 
        self.time_step = [] # total time per step

        # Retrieval parameters
        self.current_step = 0

        if self.query_set is not None:
            nq, d = self.query_set.shape
            assert d == self.dim
            assert nq > self.batch_size
            self.query_set_start_id = 0
            self.query_set_end_id = self.batch_size

    @torch.no_grad()
    def single_step(self):
        """
        Combine forward pass and (optionally) a retrieval, for single token generation.
        """
        
        """
        The incremental_state will only add the current state into the dictionary, 
            i.e., its size grows linearly with its step steps. 
        Thus, it incremental inference must start from the beginning, i.e.,
            not with a prefix of 500

        Example:
            step 510: 32.01556205749512 ms
            Shape of one layer in incremental_state:  torch.Size([1, 16, 510, 64])
            step 511: 31.69989585876465 ms
            Shape of one layer in incremental_state:  torch.Size([1, 16, 511, 64])
            step 512: 30.326128005981445 ms
            Shape of one layer in incremental_state:  torch.Size([1, 16, 512, 64])
        """
        start_step = time.time()

        # retrieve + inference
        if self.current_step % self.retrieval_interval == 0: 

            start_model_encoder_query = time.time()
            next_dummy_token = 1
            self.query_tokens = torch.tensor([[next_dummy_token] * self.retrieval_token_len] * self.batch_size).to(self.device)
            out_dict = self.model_encoder(self.query_tokens)   
            if self.query_set is None:         
                state_for_retrieval = out_dict['encoder_out'][0][0] # cls state [Batch_size, Hidden_dim]
                state_for_retrieval = state_for_retrieval.view(self.batch_size, self.dim).cpu().numpy()
            else:
                state_for_retrieval = self.query_set[self.query_set_start_id:self.query_set_end_id]
                self.query_set_start_id += self.batch_size
                self.query_set_end_id += self.batch_size
                if self.query_set_end_id > self.query_set.shape[0]:
                    self.query_set_start_id = 0
                    self.query_set_end_id = self.batch_size
            self.last_retrieval_state = state_for_retrieval
            end_model_encoder_query = time.time()
            
            start_retriever = time.time()
            # TODO: optionally get some real tokens from the retriever
            if self.request_with_lists:
                list_IDs, list_centroids = self.index_scanner.search(state_for_retrieval)
                self.retrieval_cache = self.retriever.retrieve_with_lists(state_for_retrieval, list_IDs)
            else:
                self.retrieval_cache = self.retriever.retrieve(self.last_retrieval_state)
            end_retriever = time.time()

            start_model_encoder_knowledge = time.time()
            self.encoder_out_dict = self.model_encoder(self.retrieved_tokens)
            end_model_encoder_knowledge = time.time()

            start_model_decoder = time.time()
            out_tensor, out_dict = self.model_decoder(self.input_tokens, encoder_out=self.encoder_out_dict, incremental_state=self.incremental_state)
            end_model_decoder = time.time()

            self.time_retriever.append(end_retriever - start_retriever)
            self.time_model_encoder.append(end_model_encoder_knowledge - start_model_encoder_knowledge + end_model_encoder_query - start_model_encoder_query)

        # only inference
        else: 
            start_model_decoder = time.time()
            out_tensor, out_dict = self.model_decoder(self.input_tokens, encoder_out=self.encoder_out_dict, incremental_state=self.incremental_state)
            end_model_decoder = time.time()

            self.time_retriever.append(0)
            self.time_model_encoder.append(0)
        

        # expand the input token sequence
        self.current_step += 1
        next_dummy_token = 1 # always set the next token ID as 1
        self.input_tokens = torch.tensor([[next_dummy_token] * 1] * self.batch_size).to(self.device)

        end_step = time.time()

        self.time_model_decoder.append(end_model_decoder - start_model_decoder)
        self.time_model.append(self.time_model_decoder[-1] + self.time_model_encoder[-1])
        self.time_step.append(end_step - start_step)

    @torch.no_grad()
    def multi_steps(self, num_step : Optional[int] = 1, nprobe : Optional[int] = 1):
        """
        Combine forward pass and (optionally) a retrieval, for multiple token generations.
        """
        for i in range(num_step):
            self.single_step()

    @torch.no_grad()
    def batch_inference(self, num_step : Optional[int] = 1, nprobe : Optional[int] = 1):
        """
        multi_steps with a reset_inference_state
        """
        self.reset_inference_state()
        self.time_start_batch = time.time()
        self.multi_steps(num_step, nprobe)
        self.time_end_batch = time.time()

    def get_profiling(self):
        """
        Return the following three arrays:
            model inference per step in sec 
            retrieval per step (0 if no retrieval) in sec 
            total time per step
        """
        return self.time_model, self.time_model_encoder, self.time_model_decoder, self.time_retriever, self.time_step
    
    def print_profiling_stats(self):
        """
        Print some stats of profiling
        """
        assert len(self.time_model) == len(self.time_model_encoder)
        assert len(self.time_model) == len(self.time_model_decoder)
        assert len(self.time_model) == len(self.time_retriever)
        assert len(self.time_model) == len(self.time_step)

        # in ms
        time_model = np.array(self.time_model) * 1000
        time_model_encoder = np.array(self.time_model_encoder) * 1000
        time_model_decoder = np.array(self.time_model_decoder) * 1000
        time_retriever = np.array(self.time_retriever) * 1000
        time_step = np.array(self.time_step) * 1000
        time_other = time_step - time_model - time_retriever

        time_batch = (self.time_end_batch - self.time_start_batch) * 1000
        print("== Batch profiling: ==")
        print("Batch size: {}".format(self.batch_size))
        print("Throughput (steps * batch_size / sec): {:.2f}".format(
            self.batch_size * len(time_step) / (time_batch / 1000)))
        print("Latency (ms/step): {:.2f}".format(time_batch/len(time_step)))
        print("Total time steps: {}".format(len(time_model)))
        print("Entire batch time: {:.2f} ms".format(time_batch))
        print("\n")

        print("Total time steps: {}".format(self.current_step))
        print("Step (overall): total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_step), np.average(time_step)))
        print("Inference: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_model), np.average(time_model)))
        print("\tEncoder: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_model_encoder), np.average(time_model_encoder)))
        print("\tDecoder: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_model_decoder), np.average(time_model_decoder)))
        print("Retrieval: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_retriever), np.average(time_retriever)))
        print("Other: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_other), np.average(time_other)))