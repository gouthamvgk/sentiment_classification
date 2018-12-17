import os
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = torch.load('parameters.pth')


class model_trainer():

    """
    This object is used for training and testing the model created for
    sentiment classification.
    """

    def __init__(self, model, embedding, mod_optimizer, emb_optimizer, criterion):

        """
        model: tree LSTM model.
        embedding: Embedding layer for vocabulary.
        mod_optimizer: optimizer for the tree LSTM model.
        emb_optimizer: optimizer for the embedding layer.
        criterion: Criterion for loss calculation.
        """

        self.model = model
        self.embedding = embedding
        self.mod_optimizer = mod_optimizer
        self.emb_optimizer = emb_optimizer
        self.criterion = criterion
        self.epoch = 0


    def train_epoch(self, dataset, batch_size, print_every = 10, save = False, save_path = ''):

        """
        This function carries out the training process.
        dataset: The dataset for training.
        batch_size: number of examples in each batch.
        print_every: Number of batches after which running statistics has to be printed.
        save: If True saves the model, embedding and optimizers after the epoch completion.
        save_path: Path where model has to be saved.
        """

        self.model.train()
        self.embedding.train()

        self.mod_optimizer.zero_grad()
        self.emb_optimizer.zero_grad()

        ind = torch.randperm(len(dataset))
        loss = []
        temp_loss = []
        temp = 0
        batch = 0
        start = time.time()
        print('Epoch {}..'.format(self.epoch + 1))
        """
        *********************************************************************************
        Unlike other models here multiple examples cannot be passed to the model at once
        because each one has different tree structure, so for batch processing we calculate
        the error for each example and weight it according to the batch size.
        *********************************************************************************
        """
        for i in range(len(dataset)):
            tree, sentence, _ = dataset[ind[i]]
            inp = sentence
            inp = inp.to(device)
            embed = self.embedding(inp)
            embed = embed.unsqueeze(1)
            output, error = self.model(tree, embed)
            loss.append(error.item())
            temp_loss.append(error.item())
            error = error/batch_size
            temp += 1
            error.backward()

            if (temp == batch_size):
                batch += 1
                if ((batch %print_every) == 0):
                    print('Loss for {}th batch is {}'.format(batch, sum(temp_loss)/len(temp_loss)))
                temp_loss = []
                self.mod_optimizer.step()
                self.emb_optimizer.step()
                self.mod_optimizer.zero_grad()
                self.emb_optimizer.zero_grad()
                temp = 0
        elapsed_time = time.time() - start
        print('Epoch {} completed in {:.0f}minutes {:.0f}seconds'.format(self.epoch + 1, elapsed_time//60, elapsed_time%60))

        self.epoch += 1
        if (save):
            torch.save({'model':self.model.state_dict(),
                        'embedding':self.embedding.state_dict(),
                        'mod_op': self.mod_optimizer.state_dict(),
                        'emb_op': self.emb_optimizer.state_dict()}, os.path.join(save_path, 'epoch_'+str(self.epoch) + '.pth'))

        print('Loss for this epoch is {}'.format(sum(loss)/len(loss)))


    def test(self, dataset):

        """
        This function is used to test the model on the test dataset.
        dataset:Test dataset
        """

        self.model.eval()
        self.embedding.eval()

        predictions = torch.zeros(len(dataset)).long()
        correct = torch.zeros(len(dataset)).long()

        for i in range(len(dataset)):
            tree, sentence, label = dataset[i]
            inp = sentence
            inp = inp.to(device)
            embed = self.embedding(inp)
            embed = embed.unsqueeze(1)
            correct[i] = int(label)
            output, error = self.model(tree, embed)
            _, pred = torch.max(output, 1)
            predictions[i] = int(pred.item())

        return correct, predictions
