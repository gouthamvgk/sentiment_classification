import os
import numpy as np
import glob
import torch
from model_utils import vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = torch.load('parameters.pth')


def build_vocabulary(source_files, destination_file):

    """
    This function is used to create a file that contains all the words
    in the sentiment tree bank.
    source_files: A list containing path to the test and train sentences.
    destination_file: The file to be created and to which words has to be appended.
    """

    voc = set()
    for file in source_files:
        with open(file, 'r') as temp:
            for line in temp:
                line = line.lower()
                voc |= set(line.split())

    with open(destination_file, 'w') as file:
        for word in sorted(voc):
            file.write(word + '\n')


def create_glove_vocab(path):

    """
    This function is used to create a file that containes all the words
    in the for which glove embeddings exists.
    path: The directory containing the glove file.
    """

    print('Be patient, this takes quite a bit of time...')
    with open(os.path.join(path, 'glove_vocab.txt'), 'w') as f1, open(os.path.join(path, 'glove.840B.300d.txt'), 'r') as f2:
        for line in f2:
            words = line.split()
            f1.write(words[0] + '\n')



def create_embedding_vector(data_vocab_path, glove_vocab_path, glove_path):

    """
    This function is used to create a torch tensor that contains a 300-D
    embedding for every word in the vocabulary.  If the word exists in glove
    vocab pre-training embedding is assigned, else random embedding is initialised.
    data_vocab_path: path to the file containing all words in vocabulary.
    glove_vocab_path: path to file containing all words in glove file.
    glove_path: path to the glove file.
    """

    print('Creating embedding vectors...Be patient...')
    data_vocab = vocab(data_vocab_path)
    glove_vocab = vocab(glove_vocab_path)
    data_glove_mapping = {}
    embedding = torch.zeros(len(data_vocab), 300)

    for ind, word in data_vocab.idxtolabel.items():
        find = glove_vocab.find_index(word)
        data_glove_mapping[ind] = find

    li = []
    for val in data_glove_mapping.values():
        if val is not None:
            li.append(val)
    li.sort()
    glove_value = {}

    with open(glove_path, 'r') as file:
        li_index = 0
        line_no = 0
        for line in file:
            if(li[li_index] == line_no):
                glove_value[li[li_index]] = line.rstrip().split()[1:]
                li_index += 1
                if li_index == len(li):
                    break
            line_no += 1

    for i, j in data_glove_mapping.items():
        if j is None:
            embedding[i] = torch.empty(300).normal_(-0.05, 0.05)
        else:
            embedding[i] = torch.Tensor(list(map(float, glove_value[j])))


    torch.save(embedding, os.path.join(os.path.dirname(data_vocab_path), 'vocab_embed.pth'))
    print('Embedding vectors for vocabulary created and saved')
