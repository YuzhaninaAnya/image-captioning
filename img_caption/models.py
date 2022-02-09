import os
import sys
import random
import numpy as np
import gensim
import torch
from torch import nn
import torchvision
from .utils import softmax
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    """
    Class with a pretrained EfficientNet b3 already available in PyTorch's torchvision module
    Discard the last two layers, since we only need to encode the image, and not classify it
    """
    def __init__(self, encoded_image_size=9):
        """
        :param encoded_image_size: cnn feature size of images
        """
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        efficient_net = torchvision.models.efficientnet_b3(pretrained=True)  # pretrained ImageNet EfficientNet

        modules = list(efficient_net.children())[:-2]
        self.efficient_net = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, input):
        """
        Forward propagation

        :param input: images (batch_size, 3, image_size, image_size)
        :return: encoded images (batch_size, encoded_image_size, encoded_image_size, 512)
        """
        out = self.efficient_net(input)  # (batch_size, 512, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 512, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 512)

        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder

        :param fine_tune: If true, allow the computation of gradients
        """ 
        for p in self.efficient_net.parameters():
            p.requires_grad = False

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.efficient_net.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.attn = nn.Linear(encoder_dim + decoder_dim, encoder_dim)
        self.v = nn.Linear(encoder_dim, 1)

    def forward(self, hidden, encoder_output):
        # encoder_output (batch_size, n_pixels, encoder_dim)
        # h (batch_size, decoder_dim)

        encoder_output = encoder_output.permute((1, 0, 2))  # (n_pixels, batch_size, encoder_dim)
        hidden = hidden.unsqueeze(0)  # (1, batch_size, decoder_dim)

        n_pixels = encoder_output.shape[0]
        hidden = hidden.repeat(n_pixels, 1, 1)  # (n_pixels, batch_size, decoder_dim)

        energy = self.attn(torch.cat((hidden, encoder_output), dim=2))  # (n_pixels, batch_size, encoder_dim + decoder_dim)
        energy = torch.tanh(energy)

        attn = self.v(energy)  # (n_pixels, batch_size, 1)
        attn = softmax(attn)

        return attn


class DecoderWithAttention(nn.Module):
    def __init__(self, embed_dim, word2idx, decoder_dim, encoder_dim=1536, dropout=0.5):
        """
        :param embed_dim: embedding size
        :param word2idx: word map
        :param decoder_dim: hidden size of RNN
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout probability
        """

        super(DecoderWithAttention, self).__init__()
        self.embed_dim = embed_dim
        self.word2idx = word2idx
        vocab_size = len(word2idx)
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=dropout)
        self.attention = Attention(encoder_dim, decoder_dim)
        self.rnn = nn.LSTMCell(encoder_dim + embed_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def load_pretrained_embeddings(self, word2vec: gensim.models.Word2Vec, trainable=False):
        """
        Loads embedding layer with pre-trained embeddings

        :param word2vec: pre-trained model for embeddings
        :param trainable: allow fine-tuning of embedding layer?
        """

        weights_matrix = np.zeros((self.vocab_size, self.embed_dim))

        for word, idx in self.word2idx.items():
            if word in word2vec.index2word:
                weights_matrix[idx] = word2vec.get_vector(word)
            else:
                weights_matrix[idx] = np.random.normal(scale=0.6,
                                                       size=(self.embed_dim,))

        self.embedding.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
        self.embedding.weight.requires_grad = trainable

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, n_pixels, n_out_channels)
        :return: hidden state, cell state (batch_size, decoder_dim)
        """

        encoder_out = encoder_out.mean(1)  # (batch_size, n_out_channels)
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)  # (batch_size, decoder_dim)

        return h, c

    def forward(self, encoder_output, captions, teacher_forcing_ratio=0.5):
        """
        Forward propagation

        :param encoder_output: encoded images (batch_size, image_size, image_size, n_out_channels)
        :param captions: encoded captions (batch_size, sequence_length)
        :param teacher_forcing_ratio: probability of using the real target outputs as each next input
        :returns: scores for vocabulary (sequence_length, batch_size, vocab_size)
        """

        batch_size = encoder_output.size(0)
        enc_out_channels = encoder_output.size(-1)
        captions_len = captions.size(1)

        encoder_output = encoder_output.view(batch_size, -1, enc_out_channels)  # (batch_size, n_pixels, n_out_channels)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_output)  # (batch_size, decoder_dim)

        # (sequence_length, batch_size, vocab_size)
        predictions = torch.zeros(captions_len, batch_size, self.vocab_size).to(device)

        input = captions[:, 0]  # (batch_size)

        for t in range(captions_len):
            input = input.unsqueeze(0)  # (1, batch_size)
            embeddings = self.dropout(self.embedding(input))  # (1, batch_size, embed_dim)

            attn = self.attention(h, encoder_output)
            # encoder_output = encoder_output.permute((1, 0, 2))  # (n_pixels, batch_size, encoder_dim)
            weigted_sum = torch.sum(attn * encoder_output.permute((1, 0, 2)), dim=0, keepdim=True)  # (1, batch_size, encoder_dim)
            embeddings = torch.cat((embeddings, weigted_sum), dim=2).squeeze(0)  # (batch_size, n_out_channels + emb_dim)

            h, c = self.rnn(embeddings, (h, c))  # (batch_size, decoder_dim)
            prediction = self.fc(h)  # (batch_size, vocab_size)

            predictions[t] = prediction  # (sequence_length, batch_size, vocab_size)

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(-1)  # (batch_size)

            input = captions[:, t] if teacher_force else top1

        return predictions


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image, captions, teacher_forcing_ratio=0.5):
        """
        Forward backpropagation

        :param image: (batch_size, n_channels, image_size, image_size)
        :param captions: (batch_size, sequence_length)
        :param teacher_forcing_ratio: probability of using the real target outputs as each next input
        :return:
        """

        encoder_output = self.encoder(image)
        predictions = self.decoder(encoder_output, captions, teacher_forcing_ratio)

        return predictions
