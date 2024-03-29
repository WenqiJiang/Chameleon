from random import random
import torch.nn.functional as F

# parameters
MAX_SEQUENCE_LENGTH = 8
VOCABULARY_SIZE = 8
N_ENCODERS = 1
N_DECODERS = 1

# custom types
Vector = list[float]


# helper functions
def get_random_vector(length: int) -> Vector:
    vector = [random() for _ in range(length)]
    return vector


# components
class Datastore:
    r"""
    Interface to FPGA cluster.
    Python wrapper for whatever code we have for it.
    """
    def __init__(self):
        self.vector_store = {}
    
    def __getitem__(self, name):
        return self.vector_store[name]
    
    def __setitem__(self, name, data):
        pass
    
    def __delitem__(self, name):
        pass

    def build_index(self):
        pass


class LanguageModel:
    def __init__(self, max_sequence_length: int):
        self.max_sequence_length = max_sequence_length

    def encode(self, input: str) -> Vector:
        return get_random_vector(self.max_sequence_length)


class Retriever:
    def __init__(self, datastore: Datastore):
        pass

    def get_neighbours(self, query: Vector, k: int) -> list[Vector]:

        neighbours = [get_random_vector(len(query)) for _ in range(k)]
        return neighbours


class Combiner:
    def __init__(self, lambda_: float, probability_dim, temperature: float):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim
    
    def get_knn_prob():
        pass

    def get_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs):
        neural_model_prob = F.softmax(neural_model_logit, dim=-1)
        combined_probs = knn_prob * lambda_ + neural_model_prob * (1 - lambda_)



if __name__ == "__main__":

    datastore = Datastore()

    retriever = Retriever(datastore)
    language_model = LanguageModel(MAX_SEQUENCE_LENGTH)
    combiner = Combiner()

    x = get_random_vector(4)
    print(f"x: {x}")

    y = language_model.encode(x)
    print(f"y = encoded_x: {y}")

    neighbours = retriever.get_neighbours(y, 2)
    print(f"neighbours: {neighbours}")
