import torch

from encoders.baseencoder import AbstractEncoder
from eqemb.model import *
from eqemb.globals import device
from eqemb.utils import *


class EquationEncoderWrapper(AbstractEncoder):
    def __init__(self, modelpath):
        model = torch.load(modelpath, map_location=device)
        self.encoder = model["encoder"]
        self.eqlang = model["eq_lang"]


    def get_representation_vector_size(self):
        return self.encoder.hidden_size


    @torch.no_grad()
    def get_encoding(self, data: tuple):
        prefix_eq = self.getPrefixNotation(data[1])
        eq_tensor = tensorFromEquation(self.eqlang, prefix_eq)
        _, enc_hidden = runEncoder(self.encoder, eq_tensor, None)
        return enc_hidden.squeeze(0).squeeze(0).cpu().numpy()


    def getPrefixNotation(self, tree):
        preorder = [node.name.lower() for node in tree]
        prefix = " ".join(preorder[1:])
        return prefix
