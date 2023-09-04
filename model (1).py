import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class TextCNNEncoder(nn.Module):
    """Text encoder using CNN"""
    def __init__(self, kernel_size, num_channel):
        """
        Input:
            - kernel_size: a list for size of the kernels. e.g. [3, 4, 5] means we
                will have three kernels with size 3, 4, and 5 respectively.
            - num_channel: number of output channels for each kernel.

        A few key steps of the network:
            conv -> relu -> global max pooling -> concatenate
        
        Here we construct a list of 1d convolutional networks and store them in one pytorch object
        called ModuleList. Note we have varying kernel size and padding over this list, and
        later in the forward function we can iterate over self.convs to pass data through each network
        we've just set up.
        """
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(300, num_channel, k,
            padding=k // 2) for k in kernel_size])
            
            
        
    
    def forward(self, text_emb):
        """
        Input:
            - text_emb: input utterances with shape (N, L, 300), where N is the
                number of utterances in a batch, L is the longest utterance.
                Note we concatenate utterances from all dialogues.
        Return:
            - output: encoded utterances with shape (N, len(kernel_size) * num_channel)

        The purpose of a forward function is exactly what it sounds like, here we tell
        pytorch how to pass data "forward" through our network. Pytorch will automatically
        calculate a backward function based off of this forward function.
        """

        output = None
        
        # TextCNN forward
        output = torch.transpose(text_emb, 1, 2)
        # Loop through each convolutional layer, passing our data in and then through a relu activation function
        output = [F.relu(conv(output)) for conv in self.convs]
        # Perform a max pooling over each convolutional output
        output = [i.max(dim=2)[0] for i in output]
        output = torch.cat(output, 1)

        return output


class UtteranceEmoClf(nn.Module):
    """Single utterance emotion classifier."""
    def __init__(self, kernel_size, num_channel, drop_rate):
        """
        Input:
            - kernel_size: a list for size of the kernels for CNN. e.g. [3, 4, 5]
                means we will have three kernels with size 3, 4, and 5 respectively.
            - num_channel: number of output channels for CNN.
            - drop_rate: dropout rate.
        
        A few key steps of the network:
            textcnn -> concat text_emb and audio_emb -> dropout -> fully connected layer
        """
        super().__init__()
        # TODO
        # Set up self.textcnn with a TextCNNEncoder instance

        # Calculate the cnn_out_dim

        # Set up self.linear fully connected linear layer using nn.Linear, output size 7

        # Setup self.drop dropout layer using nn.Dropout with the given drop_rate
        self.textcnn = TextCNNEncoder(kernel_size=kernel_size, num_channel=num_channel)
        self.cnn_out_dim = (1611+ num_channel) * len(kernel_size)
        self.linear = nn.Linear(1911, 7)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, text_emb, audio_emb, num_utt):
        """
        Input:
            - text_emb: input utterances with shape (N, L, 300), where N is the
                number of utterances in a batch, L is the longest utterance.
                Note we concatenate utterances from all dialogues.
            - audio_emb: audio embedding with shape (B, T, 1611), where B is
                the batch size (number of dialogues), T is sequence length
                (max number of utterances), 1611 is the number of features
                for audio embedding.
            - num_utt: list, stores the number of utterances in each dialogue.
        Return:
            - output: (B, T, 7), where 7 is the number of emotions we want to classify.
                This stores the scores for each emotion before softmax layer.
        """
        output = None

        # calculate utterance embedding using TextCNN
        text_emb = self.textcnn(text_emb)

        # Break the dialogues into utterances-level features to be concatenated to the utterance-level audio features
        # use torch.split() to split text_emb, then use pad_sequence()
        # to pad each utterance to the same length
        text_emb = torch.split(text_emb, num_utt, dim=0)
        text_emb = pad_sequence(text_emb, batch_first=True)

        # concatenate text_emb and audio_emb
        x = torch.cat((text_emb, audio_emb), dim=2)

        # apply dropout
        x = self.drop(x)

        # linear layer
        output = self.linear(x)

        return output