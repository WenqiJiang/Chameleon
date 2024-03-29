import torch

from ralm.legacy_lm.encoder import GPT
from ralm.legacy_lm.utils import ConfigDecoder

model_config = GPT.get_default_config()
model_config.update_from_dict({
    'model_type': 'gpt-nano',
    'vocab_size' : 50257, # openai's model vocabulary
    'block_size' : 1024,  # openai's model block_size
    })
model = GPT(model_config)

idx = torch.tensor([[0, 1, 2, 3, 4]])
logit, loss = model(idx)
print(f'logit: {logit}, shape: {logit.shape}')
print(f'loss: {loss}')