import time
import copy
from nltk.draw.table import Label

import torch
import torch.nn as nn
import torch.optim as optimizer
from sklearn.metrics import log_loss, recall_score, confusion_matrix, f1_score

def calculate_loss(class_weights, scores, label):
    """
    Calculate the loss.
    Input:
        - class_weights: weight for each class.
        - scores: output scores from the model.
            Tensor of shape (B, T, C), where B is the batch size (number of
            dialogues), T is sequence length (max number of utterances),
            C=7 is the number of emotions we want to classify.
        - label: true label. Tensor of shape (B, T)

    Note: DO NOT include padded utterances in loss
    Hint: padded utterances have label -1, use nn.CrossEntropyLoss as your loss function making sure
          to specify the weight and ignore_index arguments
    """
    # TODO
    
    scores = scores.permute(0,2,1)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    loss = loss_fn(scores, label)
    return loss
    
    
def get_optimizer(net, lr, weight_decay):
    """
    Return the optimizer (Adam) we will use to train the model.
    Input:
        - net: model
        - lr: initial learning_rate
        - weight_decay: weight_decay in optimizer
    """
    return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


def train_model(net, trn_loader, val_loader, optim, class_weights, num_epoch=50,
        collect_cycle=30, device='cpu', verbose=True):
    """
    Train the model
    Input:
        - net: model
        - trn_loader: dataloader for training data
        - val_loader: dataloader for validation data
        - optim: optimizer
        - class_weights: weight for each class
        - num_epoch: number of epochs to train
        - collect_cycle: how many iterations to collect training statistics
        - device: device to use
        - verbose: whether to print training details
    Return:
        - best_model: the model that has the best performance on validation data
        - stats: training statistics
    """
    # Initialize:
    # -------------------------------------
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_wf1, best_conf_mat = None, 0, None

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        ############## Training ##############
        net.train()
        for text_emb, audio_emb, label, num_utt in trn_loader:
            num_itr += 1
            text_emb, audio_emb, label = text_emb.to(device), audio_emb.to(device), label.to(device)
            
            ############ TODO: calculate loss, update weights ############
            optim.zero_grad()
    
            outputs = net(text_emb, audio_emb, num_utt)
            outputs = outputs.permute(0,2,1)
            loss = criterion(outputs, label)
            loss.backward()
            optim.step()
            
            
            
            
            ###################### End of your code ######################
            
            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
                ))

        ############## Validation ##############
        _, wf1, loss, confusion_mat = get_validation_performance(net, class_weights,
            val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation weighted macro F-1: {:.4f}".format(wf1))
        # update stats
        if wf1 > best_wf1:
            best_model = copy.deepcopy(net)
            best_wf1, best_conf_mat = wf1, confusion_mat
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'weighted_F1': best_wf1,
             'confusion_mat': best_conf_mat
    }

    return best_model, stats


def get_validation_performance(net, class_weights, data_loader, device):
    """
    Evaluate model performance.
    Input:
        - net: model
        - class_weights: weight for each class.
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
    Return:
        - uar: unweighted average recall on the data
        - w_f1: weighted macro F-1 score
        - loss: loss of the last batch
        - confusion_mat: confusion matrix of predictions on the data
    """
    net.eval()
    y_true = [] # true labels
    y_pred = [] # predicted labels

    with torch.no_grad():
        for text_emb, audio_emb, label, num_utt in data_loader:
            text_emb, audio_emb, label = text_emb.to(device), audio_emb.to(device), label.to(device)
            loss = None # loss for this batch
            pred = None # predictions for this battch

            ######## TODO: calculate loss, get predictions #########
            
            
            pred = net(text_emb, audio_emb, num_utt)
            loss = calculate_loss(class_weights, pred, label)
            pred = pred.argmax(dim =2)
            
            
            ####### You don't need to average loss across iterations #####
            
            ###################### End of your code ######################

            y_true.append(torch.flatten(label.cpu()))
            y_pred.append(torch.flatten(pred.cpu()))
    
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    uar = recall_score(y_true, y_pred, labels=list(range(7)), average='macro')
    w_f1 = f1_score(y_true, y_pred, labels=list(range(7)), average='weighted')
    confusion_mat = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    
    return uar, w_f1, loss.item(), confusion_mat


def get_hyper_parameters():
    """
    Return a list of hyper parameters that we want to search from.
    Return:
        - lr: learning_rates
        - weight_decay: weight_decays
        - hidden_dim: dimension for hidden states in GRU

    Note: it takes about 4-5 minutes to train 60 epochs on Google Colab with GPU
    """
    lr, weight_decay, hidden_dim = [], [], []

    ######## TODO: try different lr, weight_decay, hidden_dim ##########
    lr = [0.001, 0.0001, 0.00001]
    weight_decay = [0, 0.0001, 0.001]
    hidden_dim = [64, 128, 256]


    ######################### End of your code #########################

    return lr, weight_decay, hidden_dim