import os
import zipfile
import glob


params = torch.load('parameters.pth')


def load_sentences(path):

    """
    This function reads the sentences separated by '|' in sentiment tree
    bank and converts them into a full sentence, appends to the list and
    returns it.
    """

    sentences = []
    with open(os.path.join(path, 'SOStr.txt'), 'r') as file:
        for line in file:
            k = ' '.join(line.split('|'))
            sentences.append(k.strip())
    return sentences



def train_test_split(path):

    """
    This function gets the dataset instance to which each sentence
    in tree bank belongs to, appends to list and returns it.
    """

    typ = []
    with open(os.path.join(path, 'datasetSplit.txt'), 'r') as file:
        file.readline()
        for line in file:
            index, ty = line.split(',')
            typ.append(int(ty))
    return typ



def load_parents(path):

    """
    This function reads the sentences in sentiment tree bank
    represented in a tree construction format, appends to the
    list and returns it.
    """

    parents = []
    with open(os.path.join(path, 'STree.txt'), 'r') as file:
        for line in file:
            sen = ' '.join(line.split('|'))
            parents.append(sen.strip())
    return parents



def create_dictionary(path):

    """
    This function creates a mapping between each phrase in the sentiment
    tree bank and their corresponding sentiment values.  The sentiment
    values are converted to discrete form of -2, -1, 0, 1, 2.
    """

    label = []
    with open(os.path.join(path, 'sentiment_labels.txt'), 'r') as file:
        file.readline()
        for line in file:
            index, rate = line.split('|')
            rate = float(rate)
            if rate <= 0.2:
                lab = -2
            elif rate <= 0.4:
                lab = -1
            elif rate >= 0.8:
                lab = 2
            elif rate >= 0.6:
                lab = 1
            else:
                lab = 0
            label.append(lab)
    dictionary = dict()
    with open(os.path.join(path, 'dictionary.txt'), 'r') as file:
        for line in file:
            sen, index = line.split('|')
            index = int(index)
            dictionary[sen] = label[index]
    return dictionary



def create_file(path):

    """
    This function creates the train and test files from the sentiment
    tree bank. It creates the directories train, test and in each directory
    creates the file parents.txt and sentences.txt containing the tree representation
    format and sentences.
    """

    train_dir = os.path.join(path, 'train')
    os.makedirs(train_dir)
    test_dir = os.path.join(path, 'test')
    os.makedirs(test_dir)

    sentences = load_sentences(os.path.join(path, 'STB'))
    splits = train_test_split(os.path.join(path, 'STB'))
    parents = load_parents(os.path.join(path, 'STB'))

    file1 = open(os.path.join(train_dir, 'sentences.txt'), 'w')
    file2 = open(os.path.join(train_dir, 'parents.txt'), 'w')
    file3 = open(os.path.join(test_dir, 'sentences.txt'), 'w')
    file4 = open(os.path.join(test_dir, 'parents.txt'), 'w')

    for sen, par, sp in zip(sentences, parents, splits):
        if (sp == 1 or sp == 2):
            file1.write(sen + '\n')
            file2.write(par + '\n')
        elif(sp == 3):
            file3.write(sen + '\n')
            file4.write(par + '\n')

    file1.close()
    file2.close()
    file3.close()
    file4.close()
