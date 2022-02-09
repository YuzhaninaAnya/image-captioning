import os
import spacy
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image

spacy_eng = spacy.load("en")


class Vocabulary:
    def __init__(self, min_word_freq):
        """
        :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>
        """
        self.min_word_freq = min_word_freq

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, texts):
        word_freq = Counter()

        for text in texts:
            for word in self.tokenize(text):
                word_freq[word] += 1

        # Create word map
        words = [word for word in word_freq.keys() if word_freq[word] > self.min_word_freq]
        word2idx = {word: idx + 1 for idx, word in enumerate(words)}
        word2idx["<unk>"] = len(word2idx) + 1
        word2idx["<start>"] = len(word2idx) + 1
        word2idx["<end>"] = len(word2idx) + 1
        word2idx["<pad>"] = 0

        self.word2idx = word2idx

    def numericalize(self, text):
        """
        For each word in the text corresponding index token for that word form the vocab built as list
        """
        tokenized_text = self.tokenize(text)
        encoding_text = [self.word2idx["<start>"]] + \
                        [self.word2idx.get(word, self.word2idx["<unk>"]) for word in tokenized_text] + \
                        [self.word2idx["<end>"]]

        return encoding_text


class FlickrDataset(Dataset):
    """
    Custom Dataset
    """
    def __init__(self, path_to_images, path_to_caption_file, transform=None, min_word_freq=5):
        """
        :param path_to_images: path to folder with images
        :param path_to_caption_file: path to file with captions
        :param transform: transforms are common image transformations
        :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>
        """

        self.path_to_images = path_to_images
        self.data = pd.read_csv(path_to_caption_file)
        self.transform = transform

        # Get image and caption column from the dataframe
        self.imgs = self.data["image"]
        self.captions = self.data["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(min_word_freq)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.path_to_images, img_name)
        img = Image.open(img_location).convert("RGB")

        # apply the transformation to the image
        if self.transform is not None:
            img = self.transform(img)

        # convert the caption text to indexes
        encoding_caption = self.vocab.numericalize(caption)

        return {"image": img,
                "caption": torch.tensor(encoding_caption)}
