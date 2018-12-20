
# Sentiment classification using tree LSTM

## Overview

A tree like LSTM network is used to predict the sentiment scores of IMDB movie reviews dataset.  The tree like LSTM was trained using the stanford sentiment treebank which consists of constituency tree and label for each sentence.  

## Dependencies

 - Python 3.6
 - Pytorch
 - Java
 - Standford NLP parser

## Installation

All the python dependency packages can be installed with `pip install` command.
If anaconda distribution is installed then `conda install` can be used.


## Pre processing

Using the Stanford sentiment treebank, constituency tree for each sentence can be constructed.  Run `python pre_process.py` to download the Stanford parser and to setup necessary files.


## Architecture
 A tree structured LSTM network proposed in the paper '[Improved semantic representations from tree LSTMs](https://arxiv.org/pdf/1503.00075.pdf)' is used to capture the semantic meaning.  They are used to model the syntactic interpretations of sentence structure, which ordinary sequence LSTMs weren't able to do.

## Training
For training the model from terminal use `python train.py`
All the necessary files will be created and pre-trained glove embeddings are used.

## Performance
The model was trained for 15 epochs after which the loss plateaued.  It was trained for binary outputs of positive or negative statement and also for outputs of five classes. The binary model was able to attain an accuracy of 83% and the five class model was able to attain an accuracy of 45% in validation set.
  

## References
The following papers were referred:

 -[Improved semantic representations from tree LSTMs](https://arxiv.org/pdf/1503.00075.pdf)<br>
 -[Recursive deep models for semantic compositionality over a sentiment treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
