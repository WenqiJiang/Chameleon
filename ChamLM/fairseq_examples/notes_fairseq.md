# Fairseq Overview

Seems to be a fairly complicated library. It includes the most important components for sequence models such as encoder and decoder.

https://github.com/facebookresearch/fairseq

# Code Structure

The transformer components:

https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer

<img src='img_md/fairseq.models.transformer.png'>

We then have three main components. 

TransformerModelBase -> TransformerModel

TransformerEncoderBase -> TransformerEncoder

TransformerDecoderBase -> TransformerDecoder

## TransformerModel

https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_legacy.py

TransformerModel is the transformer_legacy.py file, and take both encoder and decoder as inputs. 

```python
    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args
```

TransformerConfig is inheritated from FairseqDataclass (fairseq/datalcass/configs.py), which does not support fp16 configuration...

one can download pretrained models:

```python
        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            ...
        }
```

## TransformerEncoder

https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_encoder.py

**TODO: check whether the encoder is bi-directional**

```python
class TransformerEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
```



### Encoder Output

`out_dict['encoder_out'][0]` seems to be the output state to use (Batch_size, Seq_len, Hidden_dim)

```
out_dict
{'encoder_out': [tensor([[[nan, nan, ...='cuda:0')], 'encoder_padding_mask': [tensor([[True, True,...='cuda:0')], 'encoder_embedding': [tensor([[[0., 0., 0....='cuda:0')], 'encoder_states': [], 'fc_results': [], 'src_tokens': [], 'src_lengths': [tensor([[0],
       ...rch.int32)]}
       
out_dict['encoder_out']
[tensor([[[nan, nan, ...='cuda:0')]

out_dict['encoder_out'][0].shape
torch.Size([512, 2, 1024]) -> [Seq_len, Batch_size, Hidden_dim]

out_dict['encoder_embedding']
[tensor([[[0., 0., 0....='cuda:0')]

out_dict['encoder_embedding'][0].shape
torch.Size([2, 512, 1024]) -> [Batch_size, Seq_len, Hidden_dim]
```



## TransformerDecoder

https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_decoder.py

The TransformerDecoderBase class is inherited from `FairseqIncrementalDecoder` (https://github.com/facebookresearch/fairseq/blob/3f6ba43f07a6e9e2acf957fc24e57251a7a3f55c/fairseq/models/fairseq_incremental_decoder.py#L18):

```python
    """
    Incremental decoding is a special mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for teacher forcing) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.
    Compared to the standard :class:`FairseqDecoder` interface, the incremental
    decoder interface allows :func:`forward` functions to take an extra keyword
    argument (*incremental_state*) that can be used to cache state across
    time-steps.
    """
```

TransformerDecoderBase, same init as TransformerDecoder:

```python
class TransformerDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        cfg (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
```



The forward pass reuses the k-v cache `incremental_state`:

```python
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
```

The model initialization requires some args, dictionary, and embedding tokens... 

https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_decoder.py#L456

```python
class TransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
```

The args seem to be just some k-v stores. https://github.com/facebookresearch/fairseq/blob/3f6ba43f07a6e9e2acf957fc24e57251a7a3f55c/fairseq/models/transformer/transformer_config.py#L280

```python
    @classmethod
    def from_namespace(cls, args):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = DecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, DecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = EncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, EncDecBaseConfig, "encoder", seen
                        )
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                elif safe_hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args

```



```

## Transformer Inputs

### Dictionary

https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/dictionary.py

​```python
class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)
```

The asterisk "*" is **used in Python to define a variable number of arguments**. The asterisk character has to precede a variable identifier in the parameter list.


### Input: TransformerConfig / args

According to the TransformerDecoder init, args is a dictionary that will later be turned into .

https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_decoder.py#L456

```python
class TransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
```

In the `TransformerDecoderBase`, the config is used in the following way: 

```python
class TransformerDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        cfg (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

```

I tried to run the Config to see what's the default:

```python
print(cfg)

TransformerConfig(_name=None, activation_fn='relu', dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, adaptive_input=False, encoder=EncDecBaseConfig(_name=None, embed_path=None, embed_dim=512, ffn_embed_dim=2048, layers=6, attention_heads=8, normalize_before=False, learned_pos=False, layerdrop=0, layers_to_keep=None, xformers_att_config=None), max_source_positions=1024, decoder=DecoderConfig(_name=None, embed_path=None, embed_dim=512, ffn_embed_dim=2048, layers=6, attention_heads=8, normalize_before=False, learned_pos=False, layerdrop=0, layers_to_keep=None, xformers_att_config=None, input_dim=512, output_dim=512), max_target_positions=1024, share_decoder_input_output_embed=False, share_all_embeddings=False, merge_src_tgt_embed=False, no_token_positional_embeddings=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0.0, adaptive_softmax_factor=4, layernorm_embedding=False, tie_adaptive_weights=False, tie_adaptive_proj=False, no_scale_embedding=False, checkpoint_activations=False, offload_activations=False, no_cross_attention=False, cross_self_attention=False, quant_noise=QuantNoiseConfig(_name=None, pq=0.0, pq_block_size=8, scalar=0.0), min_params_to_wrap=100000000, char_inputs=False, relu_dropout=0.0, base_layers=0, base_sublayers=1, base_shuffle=1, export=False, no_decoder_final_norm=False)
```

All **Encoder config**:

```
encoder=EncDecBaseConfig(_name=None, embed_path=None, embed_dim=512, ffn_embed_dim=2048, layers=6, attention_heads=8, normalize_before=False, learned_pos=False, layerdrop=0, layers_to_keep=None, xformers_att_config=None)
```

All **Decoder config** (has identical entries except `input_dim=1024, output_dim=1024`):

```
decoder=DecoderConfig(_name=None, embed_path=None, embed_dim=512, ffn_embed_dim=2048, layers=6, attention_heads=8, normalize_before=False, learned_pos=False, layerdrop=0, layers_to_keep=None, xformers_att_config=None, input_dim=512, output_dim=512),
```

As far as I understood from the code which utilizes these parameters, the`input_dim` and `output_dim` in the decoder are redundant and should be the same as embedding dim:
https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/transformer_layer.py

https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_config.py#L61



Instantiate input from cfg `from_namespace`:

```python
    def from_namespace(cls, args):
    				...
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = DecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, DecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = EncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, EncDecBaseConfig, "encoder", seen
                        )
            ...
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args
```



`TransformerConfig`: https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_config.py#L97



The default model by using an empty `args`:

```
=== Default Model ===

  (dropout_module): FairseqDropout()
  (embed_tokens): Embedding(10, 128, padding_idx=1)
  (project_in_dim): Linear(in_features=128, out_features=512, bias=False)
  (embed_positions): SinusoidalPositionalEmbedding()
  (layers): ModuleList(
    (0-5): 6 x TransformerDecoderLayerBase(
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
      (encoder_attn): MultiheadAttention(
        (dropout_module): FairseqDropout()
        (k_proj): Linear(in_features=512, out_features=512, bias=True)
        (v_proj): Linear(in_features=512, out_features=512, bias=True)
        (q_proj): Linear(in_features=512, out_features=512, bias=True)
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (fc1): Linear(in_features=512, out_features=2048, bias=True)
      (fc2): Linear(in_features=2048, out_features=512, bias=True)
      (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
  )
  (output_projection): Linear(in_features=512, out_features=4, bias=False)
)

```

### Decoder Output 

Decoder output (incremental inference):

For retrieval: `out_dict['inner_states'][-1]` -> [1, Batch_size, Hidden_dim]

```
out_tensor: [Batch_size, 1, Vocab_size]
out_tensor
tensor([[[-0.5234,  0.7490,  0.7560,  0.4917]],
        [[-0.3204,  0.2042,  0.3755, -1.8367]]], device='cuda:0')
torch.Size([2, 1, 4])

out_dict has two entries: {'attn': [None], 'inner_states': [tensor([...}

len(out_dict['inner_states'])
13 -> Layer_num + 1 (input embeddings)

for item in out_dict['inner_states']:
	print(item)
    
tensor([[[ 20.4039,  -0.0000, -23.3466,  ..., -40.8960,  38.7881,  -1.5737],
         [ 20.4039, -12.2607, -23.3466,  ...,  -0.0000,  38.7881,  -1.5737]]],
       device='cuda:0')
tensor([[[-0.5845,  0.2275, -0.5924,  ..., -0.7898,  2.0795, -0.2049],
         [ 0.3735, -0.2660, -0.7791,  ...,  0.0776,  1.6020, -0.1366]]],
       device='cuda:0')
       ...
tensor([[[-0.5065, -0.6339,  3.8503,  ..., -0.0151, -0.1864,  0.9715],
         [-0.8171, -0.2381,  2.5466,  ...,  0.9052, -0.2968,  1.0881]]],
       device='cuda:0')
All in shape: torch.Size([1, 2, 1024]) -> [1, Batch_size, Hidden_dim]

Since the first tensor in out_dict['inner_states'] has a quite high variance, while all other
	layers have smaller variance. It could be that the last tensor out_dict['inner_states'][-1] is the last layer state
```

Decoder output (default inference) :

```
out_tensor.shape
torch.Size([2, 500, 4]) -> (Batch_size, Seq_len, Vocab_size)
out_dict has two entries: {'attn': [None], 'inner_states': [tensor([...}

len(out_dict['inner_states'])
13 -> Layer_num + 1 (input embeddings)

out_dict['inner_states'][-1].shape
torch.Size([500, 2, 1024]) -> (Seq_len, Batch_size, Vocab_size)

```


### Examples of decoder instantiation


#### My own example

```python
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.decoder = { \
    "embed_dim" : 1024, 
    "ffn_embed_dim": 4096, 
    "layers" : 12,
    "attention_heads" : 16}
args.encoder = { \
    "embed_dim" : 1024, 
    "ffn_embed_dim": 4096, 
    "layers" : 12,
    "attention_heads" : 16}

cfg = TransformerConfig.from_namespace(args)
print(f"The config created from args: {cfg}")

# vocab to vocab ID
dictionary = Dictionary()

# Input embeddings
vocab_size = 10
dec_embs = torch.nn.Embedding(vocab_size, args.decoder["embed_dim"], dictionary.pad())

model = TransformerDecoder(args, dictionary, dec_embs)
print(model)
```

Output:

```
The config created from args: TransformerConfig(_name=None, activation_fn='relu', dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, adaptive_input=False, encoder=EncDecBaseConfig(_name=None, embed_path=None, embed_dim=1024, ffn_embed_dim=4096, layers=12, attention_heads=16, normalize_before=False, learned_pos=False, layerdrop=0, layers_to_keep=None, xformers_att_config=None), max_source_positions=1024, decoder=DecoderConfig(_name=None, embed_path=None, embed_dim=1024, ffn_embed_dim=4096, layers=12, attention_heads=16, normalize_before=False, learned_pos=False, layerdrop=0, layers_to_keep=None, xformers_att_config=None, input_dim=1024, output_dim=1024), max_target_positions=1024, share_decoder_input_output_embed=False, share_all_embeddings=False, merge_src_tgt_embed=False, no_token_positional_embeddings=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0.0, adaptive_softmax_factor=4, layernorm_embedding=False, tie_adaptive_weights=False, tie_adaptive_proj=False, no_scale_embedding=False, checkpoint_activations=False, offload_activations=False, no_cross_attention=False, cross_self_attention=False, quant_noise=QuantNoiseConfig(_name=None, pq=0.0, pq_block_size=8, scalar=0.0), min_params_to_wrap=100000000, char_inputs=False, relu_dropout=0.0, base_layers=0, base_sublayers=1, base_shuffle=1, export=False, no_decoder_final_norm=False)
TransformerDecoder(
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
```


#### RoBERTa:

```python
				# Wenqi: checkout this dec_embs dimension
        dec_embs = nn.Embedding(vocab_size, embed_dim, dictionary.pad())
        if args.share_all_embeddings or args.share_decoder_input_output_embed:
            # Note: I wasn't able to use Embedding _weight parameter to achive this sharing.
            dec_embs.weight = lm_head.weight

        decoder = TransformerDecoder(
            RobertaEncDecModel.read_args_from_roberta(roberta_enc.args),
            dictionary,
            dec_embs,
            no_encoder_attn=False,
            output_projection=lm_head,
        )
```

`pad()` In Dictionary:

```python
    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index
```

`nn.Embedding` (https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

```python
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
```