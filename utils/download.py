import os
import zipfile
import numpy as np
import glob
import urllib3
import sys
from copy import deepcopy

params = torch.load('parameters.pth')


def download_file(url, dir_path):

    """
    Downloads a file to the given directory given it's url.
    url: url of the file to be downloaded.
    dir_path: path where the file has to be saved.
    """

    file_name = url.split('/')[-1]
    path = os.path.join(dir_path, file_name)
    file = open(path, 'wb')
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    file.write(response.data)
    file.close()
    response.release_conn()


def unzip(file_path, mod_name = None, others = True):

    """
    Given a zip file this function extracts it and deletes original zip file.
    file_path: path to the zip file.
    mod_name: new name for the extracted folder.
    other: If True then renames the extracted folder with mod_name.
    """

    di = os.path.dirname(file_path)
    with zipfile.ZipFile(file_path) as folder:
        fol = folder.namelist()[0]
        folder.extractall(di)
    if others == True:
        os.rename(os.path.join(di, fol), os.path.join(di, mod_name))
    os.remove(file_path)


def download_and_unzip(directory, parser = False, tagger = False, word_vec = False, tree_bank = False):

    """
    This function is used to download and extract the four files required for sentiment classification.
    directory: base directory.
    parser: If True downloads the stanford NLP parser.
    tagger: If True downloads the stanford NLP tagger.
    word_voc: If True downloads the pre-trained glove embeddings.
    tree_bank: If True downloads the stanford sentiment tree bank.
    """

    if parser == True:
        temp = os.path.join(directory, 'lib')

        if (os.path.exists(os.path.join(temp, 'stanford_parser'))):
            print('Folder already exits for parser, check if downloaded')
        else:
            if not os.path.exists(temp):
                os.makedirs(temp)
            url = 'http://nlp.stanford.edu/software/stanford-parser-full-2015-01-29.zip'
            print('Downloading Parser')
            download_file(url, temp)
            print('Parser downloaded')
            unzip(os.path.join(temp, url.split('/')[-1]), 'stanford-parser')
            print('Parser unzipped')

    if tagger == True:
        temp = os.path.join(directory, 'lib')
        
        if (os.path.exists(os.path.join(temp, 'stanford_tagger'))):
            print('Folder already exits for tagger, check if downloaded')
        else:
            if not os.path.exists(temp):
                os.makedirs(temp)
            url = 'http://nlp.stanford.edu/software/stanford-postagger-2015-01-29.zip'
            print('Downloading tagger')
            download_file(url, temp)
            print('Tagger downloaded')
            unzip(os.path.join(temp, url.split('/')[-1]), 'stanford-tagger')
            print('Tagger unzipped')

    if word_vec == True:
        if (os.path.exists(os.path.join(directory, 'glove'))):
            print('Folder already exits for word_vec, check if downloaded')
        else:
            os.makedirs(os.path.join(directory, 'glove'))
            url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
            print('Downloading word_vectors')
            download_file(url, os.path.join(directory, 'glove'))
            print('word_vectors downloaded')
            unzip(os.path.join(os.path.join(directory, 'glove'), url.split('/')[-1]), others = False)
            print('word_vectors unzipped')

    if tree_bank == True:
        if (os.path.exists(os.path.join(directory, 'STB'))):
            print('Folder already exits for tree bank, check if downloaded')
        else:
            url = 'http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'
            print('Downloading sentiment tree bank')
            download_file(url, directory)
            print('Sentiment tree bank downloaded')
            unzip(os.path.join(directory, url.split('/')[-1]), 'STB')
            print('Sentiment tree bank unzipped')
