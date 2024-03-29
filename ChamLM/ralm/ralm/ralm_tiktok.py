"""
Tik-tok scheduling version of the forward pass of RALM, both decoder-only and encoder-decoder. 
"""

import torch 
import time
import numpy as np

from typing import Optional

@torch.no_grad()
class ralmTikTokDecoder:
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
            batch_size : inference batch size = tik = tok batch size
            retrieval_interval : retrieve every N steps
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
        for step_type in ["tik", "tok"]:
            self.model(self.input_tokens[step_type], incremental_state=self.incremental_state[step_type])
       
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
        self.input_tokens = dict()
        self.incremental_state = dict()
        self.retrieval_cache = dict()

        # three profile arrays
        self.time_model = dict() # model inference per step in sec 
        self.time_retriever = dict() # retrieval per step (0 if no retrieval) in sec 
        self.time_step = dict() # total time per step

        self.start_step = dict()
        self.end_step = dict()
        self.start_model = dict()
        self.end_model = dict()
        self.start_retriever = dict()
        self.end_retriever = dict()

        # Retrieval parameters
        self.current_step = dict()

        for step_type in ["tik", "tok"]:
            self.input_tokens[step_type] = torch.tensor([[begin_of_sentence] * start_seq_len] * self.batch_size).to(self.device)
            self.incremental_state[step_type] = dict()
            self.time_model[step_type] = [] # model inference per step in sec 
            self.time_retriever[step_type] = [] # retrieval per step (0 if no retrieval) in sec 
            self.time_step[step_type] = [] # total time per step
            self.current_step[step_type] = 0

        if self.query_set is not None:
            nq, d = self.query_set.shape
            assert d == self.dim
            assert nq > self.batch_size
            self.query_set_start_id = 0
            self.query_set_end_id = self.batch_size

    @torch.no_grad()
    def single_inference_step(self, step_type):
        """
        Forward pass, assuming this is not a retrieval-involeved step, for single token generation.

        step_type = 'tik' or 'tok'
        """
        assert step_type in ['tik', 'tok']

        start_step = time.time()

        start_model = time.time()
        out_tensor, out_dict = self.model(self.input_tokens[step_type], incremental_state=self.incremental_state[step_type])
        end_model = time.time()

        self.time_retriever[step_type].append(0)

        # TODO: optionally implement a combiner

        # expand the input token sequence
        next_dummy_token = 1 # always set the next token ID as 1
        self.input_tokens[step_type] = torch.tensor([[next_dummy_token] * 1] * self.batch_size).to(self.device)

        end_step = time.time()

        self.current_step[step_type] += 1
        self.time_model[step_type].append(end_model - start_model)
        self.time_step[step_type].append(end_step - start_step)

    @torch.no_grad()
    def single_retrieve_step_send(self, step_type):
        """
        First half of a retrieval step, i.e., send the current state to the retriever.

        step_type = 'tik' or 'tok'
        """
        assert step_type in ['tik', 'tok']
        
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
        self.start_step[step_type] = time.time()
        self.start_model[step_type] = time.time()

        out_tensor, out_dict = self.model(self.input_tokens[step_type], incremental_state=self.incremental_state[step_type])
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
        self.end_model[step_type] = time.time()

        self.start_retriever[step_type] = time.time()
        if self.request_with_lists:
            list_IDs, list_centroids = self.index_scanner.search(state_for_retrieval)
            self.retriever.retrieve_with_lists_send(state_for_retrieval, list_IDs)
        else:
            self.retriever.retrieve_send(state_for_retrieval)

    @torch.no_grad()
    def single_retrieve_step_recv(self, step_type):
        """
        Second half of a retrieval step, i.e., receive the retrieved results and update the input tokens.

        step_type = 'tik' or 'tok'
        """
        assert step_type in ['tik', 'tok']
        
        self.retrieval_cache[step_type] = self.retriever.retrieve_recv()
        self.end_retriever[step_type] = time.time()

        # expand the input token sequence
        next_dummy_token = 1 # always set the next token ID as 1
        self.current_step[step_type] += 1
        self.input_tokens[step_type] = torch.tensor([[next_dummy_token] * 1] * self.batch_size).to(self.device)
        self.end_step[step_type] = time.time()
        self.time_retriever[step_type].append(self.end_retriever[step_type] - self.start_retriever[step_type])
        self.time_model[step_type].append(self.end_model[step_type] - self.start_model[step_type])
        self.time_step[step_type].append(self.end_step[step_type] - self.start_step[step_type])

    @torch.no_grad()
    def batch_inference(self, num_step : Optional[int] = 1, nprobe : Optional[int] = 1):
        """
        multi_steps with a reset_inference_state
        """
        self.reset_inference_state()

        """
        Constraints: 
            * once the retrieve send has been retriegered, this batch cannot inference until results are back
            * the first sent request must also be received first (e.g., tik_send -> tok_send -> tik_recv -> tok_recv)
        
        No constraints:
            * tik and tok does not have to be in the same step
        
        Example:
            tik_send -> tok_retrieve -> tok_inference -> tok_inference -> tik_retreive -> tik_inference 
        """
        req_on_fly = [] # 'tik' or 'tok', at most both on fly
        self.time_start_batch = time.time()

        while self.current_step['tik'] < num_step or self.current_step['tok'] < num_step:

            for step_type in ['tik', 'tok']:
                
                if self.current_step[step_type] < num_step:
                    # retrieval step
                    if self.current_step[step_type] % self.retrieval_interval == 0:
                        # not send request yet
                        if step_type not in req_on_fly:
                            self.single_retrieve_step_send(step_type)
                            req_on_fly.append(step_type)
                        # send request but not receive yet
                        else:
                            # tik is the first request to be received and there is data in the reply buffer
                            if req_on_fly[0] == step_type:
                                if self.retriever.poll():
                                    self.single_retrieve_step_recv(step_type)
                                    req_on_fly.remove(step_type)
                    # inference step
                    else:
                        self.single_inference_step(step_type)

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
        for step_type in ['tik', 'tok']:
            assert len(self.time_model[step_type]) == len(self.time_retriever[step_type])
            assert len(self.time_model[step_type]) == len(self.time_step[step_type])
        assert len(self.time_model['tik']) == len(self.time_model['tok'])

        # in ms
        time_model = dict()
        time_retriever = dict()
        time_step = dict()
        time_other = dict()

        for step_type in ['tik', 'tok']:
            time_model[step_type] = np.array(self.time_model[step_type]) * 1000
            time_retriever[step_type] = np.array(self.time_retriever[step_type]) * 1000
            time_step[step_type] = np.array(self.time_step[step_type]) * 1000
            time_other[step_type] = time_step[step_type] - time_model[step_type] - time_retriever[step_type]
            
        time_batch = time_batch = (self.time_end_batch - self.time_start_batch) * 1000
        print("== Batch profiling: ==")
        print("Batch size: {}".format(self.batch_size))
        print("Throughput (steps * batch_size / sec): {:.2f}".format(
            self.batch_size * (len(time_step[step_type]) + len(time_step[step_type])) / (time_batch / 1000)))
        print("Latency (ms/step): {:.2f}".format(time_batch / (len(time_step[step_type]) + len(time_step[step_type]))))
        print("Total time steps: {}".format(len(time_model)))
        print("Entire batch time: {:.2f} ms".format(time_batch))
        print("\n")

        for step_type in ['tik', 'tok']:
            print(f"== {step_type} step profiling: ==")
            print("Total time steps: {}".format(len(time_model[step_type])))
            print("Step (overall): total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_step[step_type]), np.average(time_step[step_type])))
            print("Inference: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_model[step_type]), np.average(time_model[step_type])))
            print("Retrieval: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_retriever[step_type]), np.average(time_retriever[step_type])))
            print("Other: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_other[step_type]), np.average(time_other[step_type])))


@torch.no_grad()
class ralmTikTokEncoderDecoder:
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
        self.input_tokens = dict()
        self.query_tokens = dict()
        self.retrieved_tokens = dict()
        self.last_retrieval_state = dict()
        self.incremental_state = dict()
        self.retrieval_cache = dict()
        self.encoder_out_dict = dict()

        # profile arrays
        self.time_model = dict() # model inference per step in sec 
        self.time_model_encoder = dict() # model inference per step in sec (encoder)
        self.time_model_decoder = dict() # model inference per step in sec (decoder)
        self.time_retriever = dict() # retrieval per step (0 if no retrieval) in sec 
        self.time_step = dict() # total time per step

        self.start_step = dict()
        self.end_step = dict()
        self.start_model = dict()
        self.end_model = dict()
        self.start_model_encoder_query = dict()
        self.end_model_encoder_query = dict()
        self.start_model_encoder_knowledge = dict()
        self.end_model_encoder_knowledge = dict()
        self.start_model_decoder = dict()
        self.end_model_decoder = dict()
        self.start_retriever = dict()
        self.end_retriever = dict()

        # Retrieval parameters
        self.current_step = dict()

        for step_type in ["tik", "tok"]:

            self.input_tokens[step_type] = torch.tensor([[begin_of_sentence] * start_seq_len] * self.batch_size).to(self.device)
            # query tokens have identical length as each retrieved chunk, but there are k retrieved chunks
            self.query_tokens[step_type] = torch.tensor([[dummy_token] * self.retrieval_token_len] * self.batch_size).to(self.device)
            self.retrieved_tokens[step_type] = torch.tensor([[dummy_token] * self.retrieval_token_len * self.k] * self.batch_size).to(self.device)
            self.incremental_state[step_type] = dict()
            self.time_model[step_type] = [] # model inference per step in sec (encoder + decoder)
            self.time_model_encoder[step_type] = [] # model inference per step in sec (encoder)
            self.time_model_decoder[step_type] = [] # model inference per step in sec (decoder)
            self.time_retriever[step_type] = [] # retrieval per step (0 if no retrieval) in sec 
            self.time_step[step_type] = [] # total time per step
            self.current_step[step_type] = 0

        if self.query_set is not None:
            nq, d = self.query_set.shape
            assert d == self.dim
            assert nq > self.batch_size
            self.query_set_start_id = 0
            self.query_set_end_id = self.batch_size

    @torch.no_grad()
    def single_inference_step(self, step_type):
        """
        Forward pass, assuming this is not a retrieval-involeved step, for single token generation.

        step_type = 'tik' or 'tok'
        """
        assert step_type in ['tik', 'tok']
        
        start_step = time.time()
        
        start_model_decoder = time.time()
        out_tensor, out_dict = self.model_decoder(self.input_tokens[step_type], encoder_out=self.encoder_out_dict[step_type], incremental_state=self.incremental_state[step_type])
        end_model_decoder = time.time()

        # expand the input token sequence
        next_dummy_token = 1 # always set the next token ID as 1
        self.input_tokens[step_type] = torch.tensor([[next_dummy_token] * 1] * self.batch_size).to(self.device)
        end_step = time.time()

        self.time_retriever[step_type].append(0)
        self.time_model_encoder[step_type].append(0)
        self.time_model_decoder[step_type].append(end_model_decoder - start_model_decoder)
        self.time_model[step_type].append(self.time_model_encoder[step_type][-1] + self.time_model_decoder[step_type][-1])
        self.time_step[step_type].append(end_step - start_step)
        self.current_step[step_type] += 1

    @torch.no_grad()
    def single_retrieve_step_send(self, step_type):
        """
        Combine forward pass and (optionally) a retrieval, for single token generation.

        step_type = 'tik' or 'tok'
        """
        assert step_type in ['tik', 'tok']
        
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

        self.start_step[step_type] = time.time()

        self.start_model_encoder_query[step_type] = time.time()
        next_dummy_token = 1
        self.query_tokens[step_type] = torch.tensor([[next_dummy_token] * self.retrieval_token_len] * self.batch_size).to(self.device)
        out_dict = self.model_encoder(self.query_tokens[step_type])            
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
        self.last_retrieval_state[step_type] = state_for_retrieval
        self.end_model_encoder_query[step_type] = time.time()
        
        self.start_retriever[step_type] = time.time()
        # TODO: optionally get some real tokens from the retriever

        if self.request_with_lists:
            list_IDs, list_centroids = self.index_scanner.search(self.last_retrieval_state[step_type])
            self.retriever.retrieve_with_lists_send(self.last_retrieval_state[step_type], list_IDs)
        else:
            self.retriever.retrieve_send(self.last_retrieval_state[step_type])

    @torch.no_grad()
    def single_retrieve_step_recv(self, step_type):
        """
        Second half of a retrieval step, i.e., receive the retrieved results and update the input tokens.

        step_type = 'tik' or 'tok'
        """
        assert step_type in ['tik', 'tok']
        
        # TODO: optionally get some real tokens from the retriever
        self.retrieval_cache[step_type] = self.retriever.retrieve_recv()
        self.end_retriever[step_type] = time.time()

        self.start_model_encoder_knowledge[step_type] = time.time()
        self.encoder_out_dict[step_type] = self.model_encoder(self.retrieved_tokens[step_type])
        self.end_model_encoder_knowledge[step_type] = time.time()

        start_model_decoder = time.time()
        out_tensor, out_dict = self.model_decoder(self.input_tokens[step_type], encoder_out=self.encoder_out_dict[step_type], incremental_state=self.incremental_state[step_type])
        end_model_decoder = time.time()

        next_dummy_token = 1 # always set the next token ID as 1
        self.input_tokens[step_type] = torch.tensor([[next_dummy_token] * 1] * self.batch_size).to(self.device)

        self.end_step[step_type] = time.time()

        self.time_retriever[step_type].append(self.end_retriever[step_type] - self.start_retriever[step_type])
        self.time_model_encoder[step_type].append(self.end_model_encoder_query[step_type] - self.start_model_encoder_query[step_type] + self.end_model_encoder_knowledge[step_type] - self.start_model_encoder_knowledge[step_type])
        self.time_model_decoder[step_type].append(end_model_decoder - start_model_decoder)
        self.time_model[step_type].append(self.time_model_encoder[step_type][-1] + self.time_model_decoder[step_type][-1])
        self.time_step[step_type].append(self.end_step[step_type] - self.start_step[step_type])
        self.current_step[step_type] += 1


    @torch.no_grad()
    def batch_inference(self, num_step : Optional[int] = 1, nprobe : Optional[int] = 1):
        """
        multi_steps with a reset_inference_state
        """
        self.reset_inference_state()

        """
        Constraints: 
            * once the retrieve send has been retriegered, this batch cannot inference until results are back
            * the first sent request must also be received first (e.g., tik_send -> tok_send -> tik_recv -> tok_recv)
        
        No constraints:
            * tik and tok does not have to be in the same step
        
        Example:
            tik_send -> tok_retrieve -> tok_inference -> tok_inference -> tik_retreive -> tik_inference 
        """
        req_on_fly = [] # 'tik' or 'tok', at most both on fly
        self.time_start_batch = time.time()

        while self.current_step['tik'] < num_step or self.current_step['tok'] < num_step:

            for step_type in ['tik', 'tok']:
                
                if self.current_step[step_type] < num_step:
                    # retrieval step
                    if self.current_step[step_type] % self.retrieval_interval == 0:
                        # not send request yet
                        if step_type not in req_on_fly:
                            self.single_retrieve_step_send(step_type)
                            req_on_fly.append(step_type)
                        # send request but not receive yet
                        else:
                            # tik is the first request to be received and there is data in the reply buffer
                            if req_on_fly[0] == step_type:
                                if self.retriever.poll():
                                    self.single_retrieve_step_recv(step_type)
                                    req_on_fly.remove(step_type)
                    # inference step
                    else:
                        self.single_inference_step(step_type)

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
        for step_type in ['tik', 'tok']:
            assert len(self.time_model[step_type]) == len(self.time_model_encoder[step_type])
            assert len(self.time_model[step_type]) == len(self.time_model_decoder[step_type])
            assert len(self.time_model[step_type]) == len(self.time_retriever[step_type])
            assert len(self.time_model[step_type]) == len(self.time_step[step_type])
        assert len(self.time_model['tik']) == len(self.time_model['tok'])

        # in ms
        time_model = dict()
        time_model_encoder = dict()
        time_model_decoder = dict()
        time_retriever = dict()
        time_step = dict()
        time_other = dict()

        for step_type in ['tik', 'tok']:
            time_model[step_type] = np.array(self.time_model[step_type]) * 1000
            time_model_encoder[step_type] = np.array(self.time_model_encoder[step_type]) * 1000
            time_model_decoder[step_type] = np.array(self.time_model_decoder[step_type]) * 1000
            time_retriever[step_type] = np.array(self.time_retriever[step_type]) * 1000
            time_step[step_type] = np.array(self.time_step[step_type]) * 1000
            time_other[step_type] = time_step[step_type] - time_model[step_type] - time_retriever[step_type]

        time_batch = time_batch = (self.time_end_batch - self.time_start_batch) * 1000
        print("== Batch profiling: ==")
        print("Batch size: {}".format(self.batch_size))
        print("Throughput (steps * batch_size / sec): {:.2f}".format(
            self.batch_size * (len(time_step[step_type]) + len(time_step[step_type])) / (time_batch / 1000)))
        print("Latency (ms/step): {:.2f}".format(time_batch / (len(time_step[step_type]) + len(time_step[step_type]))))
        print("Total time steps: {}".format(len(time_model)))
        print("Entire batch time: {:.2f} ms".format(time_batch))
        print("\n")
        
        for step_type in ['tik', 'tok']:
            print(f"== {step_type} step profiling: ==")
            print("Total time steps: {}".format(self.current_step[step_type]))
            print("Step (overall): total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_step[step_type]), np.average(time_step[step_type])))
            print("Inference: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_model[step_type]), np.average(time_model[step_type])))
            print("\tEncoder: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_model_encoder[step_type]), np.average(time_model_encoder[step_type])))
            print("\tDecoder: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_model_decoder[step_type]), np.average(time_model_decoder[step_type])))
            print("Retrieval: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_retriever[step_type]), np.average(time_retriever[step_type])))
            print("Other: total time: {:.2f} ms\t average time: {:.2f} ms".format(np.sum(time_other[step_type]), np.average(time_other[step_type])))
