# EECS498/598 HW4
import os
import pickle
from collections import Counter

import torch
# this import is causing issues. Needs to be lowercase which clashes with variables
# replaced it with
import torchtext
#from torchtext.vocab import Vocab
from torch.utils.data import Dataset
import pandas as pd
from nltk import word_tokenize
# add this
from collections import OrderedDict
import numpy as np


def find_missing_dialogue_id(df):
    """
    Check if there are missing Dialogue_ID in df. If yes, return missing Dialogue_ID.
    Input:
        - df: pandas dataframe in which we need to find missing Dialogue_ID.
    Return:
        - ids: list of missing Dialogue_ID.
        return [] if there is no missing Dialogue_ID
    
    Hint: you can check every Dialogue_ID in the range [0, max(Dialogue_ID)], if
    it is not in df, add it to the output list.
    """
    # TODO
    results = []

    #for i in range(0, max(df["Dialogue_ID"])):
    #  item = df[i][df["Dialogue_ID"]]
    #  if(item not in df): #??
    #    results.append(item)

    #return results

    max_id = df["Dialogue_ID"].max()
    all_ids = set(range(max_id+1))
    present_ids = set(df["Dialogue_ID"].unique()) #return unique values in a column
    results = list(all_ids - present_ids)
    results.sort()

    return results


def get_class_stats(train_data, label_index):
    """
    Calculate the number of utterances for each emotion label.
    Input:
        - train_data: training data on which to get statistics.
        - label_index: indices for labels
    Return:
        - output: a list of length len(label_index) (in this case 7), that stores
            the number of utterances for each emotion label. Results should be
            in the order specified by label_index.
    """
    # TODO
    output = []
    for i in label_index.keys():
      #if(train_data[i] == label_index):
      #  output[label_index] += 1
      output.append(train_data[train_data["Emotion"] == i].shape[0])
      #ctr = train_data["Emotion"].value_counts(label_index)
      #output.append(ctr)
      
    #output.sorted()
    
    return output


def get_vocabulary(train_file, min_freq, specials=['<unk>']):
    """
    Preprocess utterances in training data and create vocabulary.
    Read train_file with pd.read_csv().
    Input:
        - train_file: filename of training data.
        - min_freq: the minimum frequency needed to inlcude a token in the
            vocabulary. e.g. set min_freq = 3 to only include tokens that
            appear at least 3 times in the training data.
    Return:
        - vocab: torchtext.vocab.Vocab instance that stores the vocabulary of
            training data.
    Preprocess steps:
        1. convert utterance to lower case
        2. tokenize utterance with nltk.word_tokenize
        3. update the counter
        4. create vocabulary using counter
    """
    # TODO
    vocab = None

    ctr = Counter()
    for sentence in pd.read_csv(train_file)["Utterance"]:
      tsentence = word_tokenize(sentence.lower())
      for word in tsentence:
        ctr.update({word:1})

    vocab = torchtext.vocab.vocab(ctr, min_freq = min_freq, specials = specials)

    return vocab


def load_glove_matrix(gloveFile, vocab):
    """
    Load the pretrained glove embedding, only keep tokens that are in our vocabulary.
    Input:
        - gloveFile: file that stores the pretrained glove embedding.
        - vocab: torchtext.vocab.Vocab instance that stores the vocabulary of the
        training data.
    Return:
        - W: torch tensor with shape (num_vocab, 300), where num_tokens is the
            size of vocabulary in the training data (with special tokens <unk>).
            Each row of W is the Glove embedding of a token, e.g. W[i] is the
            embedding for the token at index i.
        Note: use all zeros as the embedding for <unk>
    
    Note: if a token in the vocabulary does not appear in the GloVe file, its
        embedding should be zeros.
    Hint: you can use torch.zeros() to create the tensor first, and assign values
        for some rows.
    """
    # TODO
    W = torch.zeros((len(vocab), 300))
    keys = vocab.get_stoi().keys()

    df = open(gloveFile, encoding = "utf8")

    for row in df:
      values = row.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype = np.float64)
      if word in keys:
        W[vocab[word]] = torch.tensor(coefs)

    #W = torch.tensor(W)

    return W


class MELDDataset(Dataset):
    """Dataset for MELD."""

    def __init__(self, data, vocab, label_index, audio_emb, W):
        """
        - data: pandas dataframe of the data.
        - vocab: torchtext.vocab.Vocab instance that stores the vocabulary of
        training data.
        - label_index: mapping of each label to its assigned index.
        - audio_emb: dictionary of audio embedding.
        - W: W returned from load_glove_matrix().
        """
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.label_index = label_index
        self.audio_emb = audio_emb
        self.W = W
    
    def __len__(self):
        """Return the number of dialogues."""
        # TODO
        result = self.data["Dialogue_ID"].max()
        return result

    def __getitem__(self, idx):
        """
        Input:
            - idx: dialogue_id
        Return:
            - text_emb: list of torch tensors with shape (num_tokens, 300) that
                represents the text embedding of each utterance in dialogue_id,
                num_tokens is the number of tokens in that utterance.
                The length of text_emb should be the number of utterances in dialogue_id.
                Remember to take the same preprocessing steps as before.
            - audio_emb: torch tensor with shape (num_utterance, 1611),
                where num_utterance is the number of utterances in dialogue_id,
                1611 is the number of features for audio embedding of each utterance.
            - label: torch tensor with shape (num_utterance,) that stores the
                label index for each utterance.
        """
        # TODO
        text_emb, audio_emb, label = [], [], []

   
        # find utterances
        utterances = self.data.loc[self.data["Dialogue_ID"] == idx]
        #maxutteranceID = np.max(utterances["Utterance_ID"])

        #for utterance in range(maxutteranceID + 1):
        #validutterances = utterances.loc[utterances["Utterance_ID"] == utterance]
        for index, row in utterances.iterrows():
          utteranceEmb = []
          rowUtterance = word_tokenize(row["Utterance"])
          for word in rowUtterance:
            wordTensor = self.W[self.vocab[word]]
            utteranceEmb.append(wordTensor)
          utterance_tensor = torch.stack(utteranceEmb)
          text_emb.append(utterance_tensor)

        # find audio_emb

          audioUtterance = self.audio_emb[(str)(row["Dialogue_ID"]) + "_" + (str)(row["Utterance_ID"])]
          audio_emb.append(audioUtterance)

        # label

          label.append(self.label_index[row["Emotion"]])

        audio_emb = torch.stack(audio_emb)
        label = torch.tensor(label)



        
        return text_emb, audio_emb, label