import torch
import os

params = {}

params['inp_dim'] = 300
params['hid_dim'] = 150

params['base_path'] = os.getcwd()

params['vocab_path'] = os.path.join(params['base_path'], 'data_vocab.txt')
params['glove_vocab_path'] = os.path.join(params['base_path'], 'glove/glove_vocab.txt')
params['glove_path'] = os.path.join(params['base_path'], 'glove/glove.840B.300d.txt')
params['embed_path'] = os.path.join(params['base_path'], 'vocab_embed.pth')
params['save_path'] = os.path.join(params['base_path'], 'save')

params['train_dataset'] = os.path.join(params['base_path'], 'train_dataset.pth')
params['test_dataset'] = os.path.join(params['base_path'], 'test_dataset.pth')

params['dropout'] = 0.5
params['num_classes'] = 2

params['embed_lr'] = 0.1
params['model_lr'] = 0.05

params['epoch'] = 16

params['save_every'] = 5
params['batch_size'] = 25

torch.save(params, os.path.join(params['base_path'], 'parameters.pth'))
torch.save(params, os.path.join(params['base_path'], 'utils', 'parameters.pth'))
