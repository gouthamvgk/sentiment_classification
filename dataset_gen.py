import os
import numpy as np
import glob
import torch
from copy import deepcopy
from model_utils import vocab, Tree


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = torch.load('parameters.pth')


class dataset_generator():

    """
    This class is used to create the dataset which is given as input
    to the model. It returns a tree object, sentence and label when
    accessed by index.
    """

    def __init__(self, path, typ = 'train', fine_grain = False):

        """
        path: the base directory.
        typ: whether to create train or test dataset.
        fine_grain: If True output label has 5 classes->0,1,2,3,4
                    If False output label has 3 classes->0,1,2
        """

        self.vocab_path = os.path.join(path, 'data_vocab.txt')
        self.fine_grain = fine_grain

        if (typ == 'train'):
            self.file_path = os.path.join(path, 'train')
        elif(typ == 'test'):
            self.file_path = os.path.join(path, 'test')
        self.voc = vocab(self.vocab_path)

        self.trees = []
        self.sentences = []
        label = []

        sentences = self.read_sentences(os.path.join(self.file_path, 'sentences.toks'))
        trees = self.read_trees(os.path.join(self.file_path, 'parents.txt'), os.path.join(self.file_path, 'labels.txt'))

        if not fine_grain:
            for i in range(len(sentences)):
                if(trees[i].label != 1):
                    self.trees.append(trees[i])
                    self.sentences.append(sentences[i])
        else:
            self.trees = trees
            self.sentences = sentences

        for i in range(len(self.sentences)):
            label.append(self.trees[i].label)
        self.label = torch.LongTensor(label)


    def __len__(self):
        #Return number of items in dataset.
        return len(self.trees)


    def read_sentences(self, path):

        """
        Reads the sentences for each item in dataset and returns it.
        """
        sentences = []
        with open(path, 'r') as file:
            for line in file:
                line = line.rstrip().lower().split()
                values = self.voc.convert_to_index(line, None)
                values = torch.LongTensor(values)
                sentences.append(values)
        return sentences


    def __getitem__(self, index):

        """
        Provides indexed access of the dataset object.
        """
        tree = deepcopy(self.trees[index])
        sentence = deepcopy(self.sentences[index])
        label = deepcopy(self.label[index])

        return tree, sentence, label


    def read_trees(self, parent_path, label_path):

        """
        Creates a tree object for all items in the dataset
        and returns it.
        """
        trees = []
        parent_file = open(parent_path, 'r')
        label_file = open(label_path, 'r')
        parent = parent_file.readlines()
        label = label_file.readlines()
        for p, l in zip(parent, label):
            temp = self.read_tree(p.rstrip(), l.rstrip())
            trees.append(temp)
        return trees

    def parse_token(self, value):
        if value == '#':
            return None
        if value == str(None):
            return 1
        if self.fine_grain:
            return int(value) + 2
        else:
            if int(value) < 0:
                return 0
            elif int(value) == 0:
                return 1
            elif int(value) > 0:
                return 2

    def read_tree(self, par, lab):

        """
        Creates a tree for a example using it's label and parent values.
        """
        parents = list(map(int,par.split()))
        trees = dict()
        root = None
        labels = list(map(self.parse_token, lab.split()))
        for i in range(1,len(parents)+1):
            if i not in trees.keys() and parents[i-1]!=-1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx] = tree
                    tree.idx = idx
                    tree.label = labels[idx-1]
                    if parent in trees.keys():
                        trees[parent].add_child(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root
