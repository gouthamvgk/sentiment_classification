import os
import glob
import sys

params = torch.load('parameters.pth')



class constituency_tree():

    """
    A basic constituency_tree node which containes
    left and right child if exists.
    """

    def __init__(self):
        self.left = None
        self.right = None


    def size(self):
        self.size =1
        if self.left is not None:
            self.size += self.left.size()
        if self.right is not None:
            self.size += self.right.size()

        return self.size


    def set_spans(self):
        if self.word is not None:
            self.span = self.word
            return self.span
        else:
            self.span = self.left.set_spans()
            if self.right is not None:
                self.span += ' ' + self.right.set_spans()

            return self.span


    def get_labels(self, spans, labels, dictionary):
        if self.span in dictionary:
            spans[self.idx] = self.span
            labels[self.idx] = dictionary[self.span]
            if self.left is not None:
                self.left.get_labels(spans, labels, dictionary)
            if self.right is not None:
                self.right.get_labels(spans, labels, dictionary)



def load_constituency_tree(parents, words):

    """
    This function creates the whole constituency_tree for a sentences
    by using the constituency_tree object.
    """

    trees = []
    root = None
    size = len(parents)
    for i in range(size):
        trees.append(None)

    word_idx = 0
    for i in range(size):
        if not trees[i]:
            idx = i
            prev = None
            prev_idx = None
            word = words[word_idx]
            word_idx += 1
            while True:
                tree = constituency_tree()
                parent = parents[idx] - 1
                tree.word, tree.parent, tree.idx = word, parent, idx
                word = None
                if prev is not None:
                    if tree.left is None:
                        tree.left = prev
                    else:
                        tree.right = prev
                trees[idx] = tree
                if parent >= 0 and trees[parent] is not None:
                    if trees[parent].left is None:
                        trees[parent].left = tree
                    else:
                        trees[parent].right = tree
                    break
                elif parent == -1:
                    root = tree
                    break
                else:
                    prev = tree
                    prev_idx = idx
                    idx = parent
    return root



def load_trees(path):

    """
    This function is used to create the constituency_trees given the path of the
    train or test directory.
    """

    constituency_trees, tokens = [], []
    with open(os.path.join(path, 'parents.txt'),'r') as pfile, open(os.path.join(path, 'sentences.txt'), 'r') as tfile:
        parents = []
        for line in pfile:
            parents.append(list(map(int, line.rstrip().split())))

        for line in tfile:
            tokens.append(line.rstrip().split())

        for k in range(len(tokens)):
            constituency_trees.append(load_constituency_tree(parents[k], tokens[k]))

        return constituency_trees, tokens



def create_label_file(path, dictionary):

    """
    This function creates the label file for test and train
    directory.
    dictionary: A dictionary object containing the mapping from phrase to
                it's sentiment score.
    """

    print('Creating label file for {} set'.format(path.split('/')[-1]))

    with open(os.path.join(path, 'labels.txt'), 'w') as labels:
        constituency_trees, tokens = load_trees(path)

        for i in range(len(constituency_trees)):
            constituency_trees[i].set_spans()

            span, label = [], []
            for j in range(constituency_trees[i].size()):
                span.append(None)
                label.append(None)
            constituency_trees[i].get_labels(span, label, dictionary)
            labels.write(' '.join(map(str, label)) + '\n')
    print('Label file created for {} set'.format(path.split('/')[-1]))
