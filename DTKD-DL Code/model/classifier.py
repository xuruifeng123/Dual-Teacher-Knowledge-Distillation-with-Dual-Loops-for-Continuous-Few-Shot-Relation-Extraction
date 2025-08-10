from torch import nn, optim
from model.base_model import base_model
import torch


class Softmax_Layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, input_size, num_class):
        """
        Args:
            num_class: number of classes
        """
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=False)
        self.fc1 = nn.Linear(self.num_class*2, self.num_class, bias=False)
    def forward(self, input):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """

        logits = self.fc(input)
        return logits


class Softmax_Layer1(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, input_size, num_class):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(Softmax_Layer1, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=False)
        self.fc1 = nn.Linear(self.num_class * 2, self.num_class, bias=False)

    def forward(self, input):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """

        logits = self.fc(input)
        return logits