class ConfigLM:

    def __init__(self):
        # A set of allowed config 
        self.allowed_keys = {} 
        
    def print(self):
        """ need to have a helper to support nested indentation for pretty printing """
        for k, v in self.__dict__.items():
            print("%s: %s\n" % (k, v))

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v for k, v in self.__dict__.items() }

    def update_from_kwargs(self, **kwargs):
        """ Update config from dict """
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys)
        
    def update_from_dict(self, d):
        """ Update config from dict """
        self.__dict__.update((k, d[k]) for k in d if k in self.allowed_keys)

class ConfigEncoder(ConfigLM):
    """ TODO: Placeholder """
    def __init__(self, **kwargs):
        raise NotImplementedError

class ConfigDecoder(ConfigLM):

    """
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C
    """
    def __init__(self):

        self.allowed_keys = {
            'model_type',
            'n_layer',
            'n_head',
            'n_embd',
            'vocab_size',
            'block_size',
            'embd_pdrop',
            'resid_pdrop',
            'attn_pdrop',
        }

class ConfigEncoderDecoder(ConfigLM):
    """ TODO: Placeholder """
    def __init__(self, **kwargs):
        raise NotImplementedError