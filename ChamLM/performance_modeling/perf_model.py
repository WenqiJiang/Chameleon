import numpy as np

parameters = {
	# Vector search parameters
    "nlist" : 32768, # number of inverted lists
    "nprobe" : 32, # number of inverted lists to probe
    "dbsize": 1e9, # number of vectors in the database
    "m" : 16, # number of subquantizers
    
	# LLM parameters
	"batch_size" : 32,
	"dim" : 512,
	"layers_encoder" : 2,
	"layers_decoder" : 24,
	"attention_heads" : 8,
	"k" : 10,
	"retrieval_interval" : 1,
	"retrieval_token_len" : 16,
	"seq_len" : 512,
	"vocab_size" : 50000,
	"model_type" : "encoder-decoder",
    
    # FPGA 
	"num_FPGA" : 1,
    "pq_bandwidth" : 32 * 1e9, # Bytes/s, per FPGA
	
	# CPU
	"num_CPU" : 8, # number of CPU cores
    "ivf_bandwidth" : 2 * 1e9, # Bytes/s, per CPU core

	# GPU
	"num_GPU" : 1, # number of GPUs
	"GPU_flops" : 10 * 1e12, # FLOPS, per GPU
	"GPU_bandwidth" : 900 * 1e9, # Bytes/s, per GPU
}

BYTES_FLOAT = 4

def get_FPGA_throughput():
    
    bytes_per_query = parameters["nprobe"] / parameters["nlist"] * parameters["dbsize"] * parameters["m"]
    bandwidth = parameters["pq_bandwidth"] * parameters["num_FPGA"]
    qps = bandwidth / bytes_per_query
    
    return qps

def get_CPU_throughput():
    
	bytes_per_query = parameters["nlist"] * parameters["dim"] * BYTES_FLOAT
	bandwidth = parameters["ivf_bandwidth"] * parameters["num_CPU"]
	qps = bandwidth / bytes_per_query
        
	return qps

def get_GPU_IVF_throughput(precomputed=True):
	"""
	Use roofline model to predict the GPU throughput

	the L2 distance computation can be factorized to three matrix multiplications:
		q^2 + 2q(IVF)^T + (IVF)^2 (can be precomputed)
		(nq, d) x (d, nq) + (nq, d) x (d, nlist) + (nlist, dim) x (dim, nlist)
	"""
	flop = 2 * (parameters["batch_size"] * parameters["dim"] * parameters["batch_size"] + \
	    parameters["batch_size"] * parameters["dim"] * parameters["nlist"])
	if not precomputed:
		flop += 2 * (parameters["nlist"] * parameters["dim"] * parameters["nlist"])

	# each matrix is loaded three times
	data_movement = 3 * BYTES_FLOAT * (parameters["batch_size"] * parameters["dim"] + parameters["nlist"] * parameters["dim"])

	qps_flop = parameters["batch_size"] * parameters["GPU_flops"] / flop
	qps_bandwidth = parameters["batch_size"] * parameters["GPU_bandwidth"] / data_movement

	print("FLOP qps: {}, Bandwidth qps: {}".format(qps_flop, qps_bandwidth))
	if qps_flop < qps_bandwidth:
		print("FLOP bound")
		qps = qps_flop
	else:
		print("Bandwidth bound")
		qps = qps_bandwidth

	return qps

def get_model_parameter_num():
	"""
	Return the number of parameters in the model

	model_type: 'decoder' or 'encoder-decoder'

Decoder Example:

	  (dropout_module): FairseqDropout()
  (embed_tokens): Embedding(10000, 512, padding_idx=1)
  (embed_positions): SinusoidalPositionalEmbedding()
  (layers): ModuleList(
    (0-11): 12 x TransformerDecoderLayerBase(
      (dropout_module): FairseqDropout()
      (self_attn): MultiheadAttention(
        (dropout_module): FairseqDropout()
        (k_proj): Linear(in_features=512, out_features=512, bias=True)
        (v_proj): Linear(in_features=512, out_features=512, bias=True)
        (q_proj): Linear(in_features=512, out_features=512, bias=True)
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (activation_dropout_module): FairseqDropout()
      (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (fc1): Linear(in_features=512, out_features=2048, bias=True)
      (fc2): Linear(in_features=2048, out_features=512, bias=True)
      (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
  )
  (output_projection): Linear(in_features=512, out_features=4, bias=False)
)

Encoder / Decoder Example: Note that the Decoder has additional cross-attention layers

Model Encoder: TransformerEncoder(
  (dropout_module): FairseqDropout()
  (embed_tokens): Embedding(10, 1024, padding_idx=1)
  (embed_positions): SinusoidalPositionalEmbedding()
  (layers): ModuleList(
    (0-11): 12 x TransformerEncoderLayerBase(
      (self_attn): MultiheadAttention(
        (dropout_module): FairseqDropout()
        (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
      )
      (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout_module): FairseqDropout()
      (activation_dropout_module): FairseqDropout()
      (fc1): Linear(in_features=1024, out_features=4096, bias=True)
      (fc2): Linear(in_features=4096, out_features=1024, bias=True)
      (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
  )
)
Model Decoder: TransformerDecoder(
  (dropout_module): FairseqDropout()
  (embed_tokens): Embedding(10, 1024, padding_idx=1)
  (embed_positions): SinusoidalPositionalEmbedding()
  (layers): ModuleList(
    (0-11): 12 x TransformerDecoderLayerBase(
      (dropout_module): FairseqDropout()
      (self_attn): MultiheadAttention(
        (dropout_module): FairseqDropout()
        (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
      )
      (activation_dropout_module): FairseqDropout()
      (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (encoder_attn): MultiheadAttention(
        (dropout_module): FairseqDropout()
        (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
      )
      (encoder_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (fc1): Linear(in_features=1024, out_features=4096, bias=True)
      (fc2): Linear(in_features=4096, out_features=1024, bias=True)
      (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
  )
  (output_projection): Linear(in_features=1024, out_features=4, bias=False)
)
	"""
	assert parameters["model_type"] in ['decoder', 'encoder-decoder']

	n_params = 0
	# Input layer & output layer (seems by default the output layer is not counted)
	n_params += parameters["dim"] * parameters["vocab_size"] 

	# Middle layers
	per_layer_qkvp = parameters["dim"] * parameters["dim"] * 4
	per_layer_ffn = 2 * parameters["dim"] * parameters["dim"] * 4
	n_params += parameters['layers_decoder'] * (per_layer_qkvp + per_layer_ffn) 
	
	# maybe we can share embedding across encoder and decoder, but not necessarily
	if parameters["model_type"] == 'encoder-decoder':
		n_params += parameters["layers_decoder"] * per_layer_qkvp
		n_params += parameters['layers_encoder'] * (per_layer_qkvp + per_layer_ffn) 
		n_params += parameters["dim"] * parameters["vocab_size"] 

	return n_params


print("FPGA qps: ", get_FPGA_throughput())
print("CPU qps: ", get_CPU_throughput())
print("GPU IVF qps: ", get_GPU_IVF_throughput())
print("Model parameter num: {} M".format(get_model_parameter_num() / 1e6))