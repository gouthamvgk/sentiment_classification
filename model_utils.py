import os


class vocab():

    """
    This creates a vocabulary object with various
    functions given the path to a file containing
    all the words in vocabulary.
    """

    def __init__(self, file):

        """
        file:path to the text vocabulary file.
        self.idxtolabel : Contains a mapping from index to each word.
        self.labeltoidx: Contains a mapping from each word to index.
        """

        self.idxtolabel = dict()
        self.labeltoidx = dict()
        self.load_file(file)


    def __len__(self):
        return len(self.labeltoidx)


    def load_file(self, path):
        with open(path, 'r') as file:
            for line in file:
                word = line.rstrip()
                self.add_word(word)

    def add_word(self, word):

        """
        Adds a word to the mappings if it doesn't already exists.
        word: word to be added to the vocabulary.
        """

        word = word.lower()
        if word in self.labeltoidx:
            index = self.labeltoidx[word]
        else:
            index = len(self.labeltoidx)
            self.labeltoidx[word] = index
            self.idxtolabel[index] = word


    def find_label(self, index, default = None):
        """
        Given a index returns the corresponding word if it
        exists in the vocab otherwise returns default.
        """
        try:
            return self.idxtolabel[index]
        except KeyError:
            return default


    def find_index(self, label, default = None):
        """
        Given a word return it's index if it exists in the
        vocab otherwise returns default.
        """
        try:
            return self.labeltoidx[label]
        except KeyError:
            return default


    def convert_to_index(self, label, unknown, start = None, stop = None):

        """
        Given a list of words it return the corresponding list of indices.
        label: list of words.
        unknown: word to used for instances not in vocabulary.
        start: If given added to front of the list.
        stop: If given added to the end of the list.
        """

        conv = []
        if start is not None:
            conv += [self.find_index(start)]

        unk_index = self.find_index(unknown)

        conv += [self.find_index(lab, unk_index) for lab in label]

        if stop is not None:
            conv += [self.find_index(stop)]

        return conv


    def convert_to_label(self, index, stop = None):

        """
        Given a list of indices it return a list of corresponding words.
        index: list of indices.
        stop: instance at which conversion has to be stopped.
        """

        words = []
        for ind in index:
            words.append(self.find_label(ind))
            if ind == stop:
                break
        return words




class Tree():

    """
    A tree object(node) which is constructed for each sentence and given as
    input to the model.
    """

    def __init__(self):
        self.parent = None #points to parent node if exists
        self.num_child = 0 #contains number of child.
        self.children = [] #contains pointers to the child node.
        self.label = None #contains the sentiment score.
        self.output = None #used in model processing.

    def add_child(self, child):
        child.parent = self
        self.num_child += 1
        self.children.append(child)

    def size(self):

        """
        Returns the size of the tree under the node.
        """
        if getattr(self, size):
            return self.size
        else:
            size = 0
            for child in self.children:
                size += child.size()
            self.size = size
            return self.size

    def height(self):

        """
        Returns the height of the node.
        """
        if getattr(self, height):
            return self.height
        else:
            height = 0
            if self.num_child > 0:
                for child in self.children:
                    if child.height() > height:
                        height = child.height()
                height += 1

            self.height = height
            return self.height
