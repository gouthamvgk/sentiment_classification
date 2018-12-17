import os
import numpy as np
import glob
import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = torch.load('parameters.pth')


class btlstm_leaf(nn.Module):

    """
    Module for leaf node in the constituency_tree_lstm.
    It has input which is the embedding vector and has no
    initial hidden state.
    """

    def __init__(self, inp_dim, hid_dim):

        """
        inp_dim: Input dimension.
        hid_dim : Number of nodes in hidden layer.
        """

        super(btlstm_leaf, self).__init__()
        self.hid_dim = hid_dim
        self.inp_dim = inp_dim

        self.inp_gate = nn.Linear(inp_dim, hid_dim)
        self.out_gate = nn.Linear(inp_dim, hid_dim)
        self.upd_gate = nn.Linear(inp_dim, hid_dim)


    def forward(self, embedding):
        inp = torch.sigmoid(self.inp_gate(embedding))
        out = torch.sigmoid(self.out_gate(embedding))
        upd = torch.tanh(self.upd_gate(embedding))

        cell_state = inp * upd
        hidd_state = out * torch.sigmoid(cell_state)

        return cell_state, hidd_state



class btlstm_non_leaf(nn.Module):

    """
    Module for non-leaf node in constituency_tree lstm.
    It has no input vector but only the intial hidden
    and cell state vectors from it's left and right child.
    """

    def __init__(self, inp_dim, hid_dim):

        """
        inp_dim: Embedding vector dimension.
        hid_dim: Number of nodes in hidden layer.
        """

        super(btlstm_non_leaf, self).__init__()
        self.hid_dim = hid_dim
        self.inp_dim = inp_dim

        self.inp_ln = nn.Linear(self.hid_dim, self.hid_dim)
        self.inp_rn = nn.Linear(self.hid_dim, self.hid_dim)

        self.fo_l_l = nn.Linear(self.hid_dim, self.hid_dim)
        self.fo_l_r = nn.Linear(self.hid_dim, self.hid_dim)
        self.fo_r_l = nn.Linear(self.hid_dim, self.hid_dim)
        self.fo_r_r = nn.Linear(self.hid_dim, self.hid_dim)

        self.out_ln = nn.Linear(self.hid_dim, self.hid_dim)
        self.out_rn = nn.Linear(self.hid_dim, self.hid_dim)

        self.upd_ln = nn.Linear(self.hid_dim, self.hid_dim)
        self.upd_rn = nn.Linear(self.hid_dim, self.hid_dim)



    def forward(self, l_cell, l_hid, r_cell, r_hid):
        inp_gate = torch.sigmoid(self.inp_ln(l_hid) + self.inp_rn(r_hid))

        out_gate = torch.sigmoid(self.out_ln(l_hid) + self.out_rn(r_hid))

        upd_gate = torch.tanh(self.upd_ln(l_hid) + self.upd_rn(r_hid))

        fo_l_gate = torch.sigmoid(self.fo_l_l(l_hid) + self.fo_l_r(r_hid))
        fo_r_gate = torch.sigmoid(self.fo_r_l(l_hid) + self.fo_r_r(r_hid))

        cell_state = (inp_gate * upd_gate) + (fo_l_gate * l_cell) + (fo_r_gate * r_cell)
        hidd_state = out_gate * torch.tanh(cell_state)

        return cell_state, hidd_state



class binary_tree_lstm(nn.Module):

    """
    Module for the whole constituency_tree lstm.
    It has leaf and non-leaf as it's sub modules.
    """

    def __init__(self, inp_dim, hid_dim, loss_criterion):

        """
        inp_dim: Embedding_vector dimension.
        hid_dim: Number of nodes in hidden layer.
        loss_criterion: criterion object for calculating loss.
        self.output_score: Used to find the label for each node in the tree.
        """

        super(binary_tree_lstm, self).__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim

        self.leaf_net = btlstm_leaf(inp_dim, hid_dim)
        self.non_leaf_net = btlstm_non_leaf(inp_dim, hid_dim)
        self.criterion = loss_criterion
        self.output_score = None

    def forward(self, sen_tree, embeddings):
        loss = 0

        if (sen_tree.num_child == 0):
            sen_tree.temp = self.leaf_net(embeddings[sen_tree.idx - 1])
        else:
            for k in range(sen_tree.num_child):
                _, child_loss = self.forward(sen_tree.children[k], embeddings)
                loss += child_loss
            left_c, left_h, right_c, right_h = self.get_child_state(sen_tree)
            sen_tree.temp = self.non_leaf_net(left_c, left_h, right_c, right_h)

        output = self.output_score(sen_tree.temp[1])
        sen_tree.output = output
        if sen_tree.label != None:
            target = torch.LongTensor([sen_tree.label])
            target = target.to(device)
            loss += self.criterion(output, target)

        return sen_tree.output, loss


    def get_child_state(self, sen_tree):

        """
        Given a node returns the hidden and cell state of it's
        left and right child.
        """

        left_c, left_h = sen_tree.children[0].temp
        right_c, right_h = sen_tree.children[1].temp

        return left_c, left_h, right_c, right_h



class const_btree_lstm(nn.Module):

    """
    This contains the whole tree lstm module with output score layer.
    """
    def __init__(self, inp_dim, hid_dim, criterion, dropout = 0.5, fine_grain = False):

        """
        dropout: dropout layer hyperparameter.
        fine_grain: If True number of classes is 5, else 3.
        """

        super(const_btree_lstm, self).__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        if (fine_grain):
            self.num_classes = 5
        else:
            self.num_classes = 3
        self.criterion = criterion
        self.model = binary_tree_lstm(inp_dim, hid_dim, criterion)
        self.model.output_score = nn.Sequential(nn.Dropout(p = dropout),
                                               nn.Linear(hid_dim, self.num_classes),
                                               nn.LogSoftmax(dim=1))


    def forward(self, tree, embeddings):
        tree_output, loss = self.model(tree, embeddings)
        return tree_output, loss
