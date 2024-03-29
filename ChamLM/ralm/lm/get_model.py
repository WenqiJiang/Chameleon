import argparse
import torch 

from typing import Optional
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.data import Dictionary

def createTransformerDecoder(args=None, dictionary=None, dec_emb=None, no_encoder_attn=True, vocab_size=50000):
    """
    Create a transformer Decoder model. User can either pass in specific configurations or
        use the default model.

    Input arguments:
        all configurations in TransformerDecoder (args, dictionary, dec_embs)
    """
    if args is None:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.decoder = { \
            "embed_dim" : 1024, 
            "ffn_embed_dim": 4096, 
            "layers" : 12,
            "attention_heads" : 16}

    if dictionary is None:
        # vocab to vocab ID
        dictionary = Dictionary()

    if dec_emb is None:
        # Input embeddings
        dec_emb = torch.nn.Embedding(vocab_size, args.decoder["embed_dim"], dictionary.pad())

    model = TransformerDecoder(args, dictionary, dec_emb, no_encoder_attn=no_encoder_attn)
    print(model)

    return model

def createTransformerEncoder(args=None, dictionary=None, dec_emb=None, no_encoder_attn=True, vocab_size=50000):
    """
    Create a transformer Encoder model. User can either pass in specific configurations or
        use the default model.

    Input arguments:
        all configurations in TransformerDecoder (args, dictionary, dec_embs)
    """
    if args is None:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.encoder = { \
            "embed_dim" : 1024, 
            "ffn_embed_dim": 4096, 
            "layers" : 12,
            "attention_heads" : 16}

    if dictionary is None:
        # vocab to vocab ID
        dictionary = Dictionary()

    if dec_emb is None:
        # Input embeddings
        dec_emb = torch.nn.Embedding(vocab_size, args.encoder["embed_dim"], dictionary.pad())

    model = TransformerEncoder(args, dictionary, dec_emb)
    print(model)

    return model

def createTransformerEncoderDecoder(
    args=None, enc_dictionary=None, enc_emb=None, dec_dictionary=None, dec_emb=None, vocab_size=50000):
    """
    Create a transformer Encoder-Decoder model. User can either pass in specific configurations or
        use the default model.
    """
    if args is None:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.encoder = { \
            "embed_dim" : 1024, 
            "ffn_embed_dim": 4096, 
            "layers" : 12,
            "attention_heads" : 16}
        args.decoder = { \
            "embed_dim" : 1024, 
            "ffn_embed_dim": 4096, 
            "layers" : 12,
            "attention_heads" : 16}

    model_encoder = createTransformerEncoder(args=args, dictionary=enc_dictionary, dec_emb=enc_emb, vocab_size=vocab_size)
    model_decoder = createTransformerDecoder(args=args, dictionary=dec_dictionary, dec_emb=dec_emb, no_encoder_attn=False, vocab_size=vocab_size)

    return model_encoder, model_decoder
    