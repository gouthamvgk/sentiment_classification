import os
import numpy as np
import glob
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import time

from model_utils import vocab
from create_vocab_embed import build_vocabulary, create_glove_vocab, create_embedding_vector
from dataset_gen import dataset_generator
from train_object import model_trainer
from model import btlstm_leaf, btlstm_non_leaf, binary_tree_lstm, const_btree_lstm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


params = torch.load('parameters.pth')


print('Make sure that pre_process.py has been executed before this, otherwise quit and execute it...')


def calculate_accuracy(vec1, vec2):
    su = (vec1==vec2).sum()
    accuracy = float(su)/vec1.size(0)
    return accuracy



train_dir = os.path.join(params['base_path'], 'train')
test_dir = os.path.join(params['base_path'], 'test')


if (params['num_classes'] == 5):
    fine_grain = True
else:
    fine_grain = False



if(os.path.exists(params['vocab_path'])):
    vocb = vocab(params['vocab_path'])
    print('Loaded the dataset vocabulary object\n')
else:
    h =glob.glob(os.path.join(params['base_path'], '*/sentences.txt'))
    build_vocabulary(h, os.path.join(params['base_path'], 'data_vocab.txt'))
    vocb = vocab(params['vocab_path'])
    print('Created and loaded the dataset vocabulary object')



if not os.path.exists(params['glove_vocab_path']):
    create_glove_vocab(os.path.join(params['base_path'], 'glove'))
    print('Vocabulary file for glove vectors created\n')



if(os.path.exists(params['embed_path'])):
    embed_vector = torch.load(params['embed_path'])
    embed_vector = embed_vector.float()
    embed_vector = embed_vector.to(device)
    print('Pre trained embedding vectors loaded\n')
else:
    create_embedding_vector(params['vocab_path'], params['glo_vocab_path'], params['glove_path'])
    embed_vector = torch.load(params['embed_path'])
    print('Pre trained embedding vectors created and loaded\n')



if(os.path.exists(params['train_dataset'])):
    train_dataset = torch.load(params['train_dataset'])
    print('Training dataset loaded with {} entries\n'.format(len(train_dataset)))
else:
    train_dataset = dataset_generator(path = params['base_path'], typ = 'train', fine_grain = fine_grain)
    torch.save(train_dataset, params['train_dataset'])
    print('Training dataset created and loaded with {} entries\n'.format(len(train_dataset)))



if(os.path.exists(params['test_dataset'])):
    test_dataset = torch.load(params['test_dataset'])
    print('Test dataset loaded with {} entries\n'.format(len(test_dataset)))
else:
    test_dataset = dataset_generator(path = params['base_path'], typ = 'test', fine_grain = fine_grain)
    torch.save(test_dataset, params['test_dataset'])
    print('Test dataset created and loaded with {} entries\n'.format(len(test_dataset)))



criterion = nn.CrossEntropyLoss()

model = const_btree_lstm(params['inp_dim'], params['hid_dim'], criterion, params['dropout'], fine_grain = fine_grain)
model = model.to(device)
print('Sentiment model created\n')


embed_layer = nn.Embedding(len(vocb), params['inp_dim'])
embed_layer = embed_layer.to(device)
embed_layer.weight.data.copy_(embed_vector)
print('Embedding layer created and pre trained weights copied\n')


mod_optimizer = optim.Adagrad(model.parameters(), lr = params['model_lr'])
embed_optimizer = optim.SGD(embed_layer.parameters(), lr = params['embed_lr'])


trainer = model_trainer(model, embed_layer, mod_optimizer, embed_optimizer, criterion)

evaluate_every = 2
print_every = 40

print('Starting training...\n')
for epoch in range(5,params['epoch']):
    if ((epoch+1) % params['save_every'] == 0):
        save = True
        path = params['save_path']
    else:
        save = False
        path = ''

    trainer.train_epoch(train_dataset, params['batch_size'], print_every = print_every, save = save, save_path = path)

    if ((epoch+1) % evaluate_every == 0):
        print('Evaluaring on test dataset')
        cor, pred = trainer.test(test_dataset)
        accuracy = calculate_accuracy(cor, pred)

        print('Accuracy for the dataset is {}'.format(accuracy))
