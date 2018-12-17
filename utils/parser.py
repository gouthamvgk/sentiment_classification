import os
import sys


params = torch.load('parameters.pth')

"""
*************************************************************************************************
Both these functions have not been used. It was used for experiment.
dependency_parse() is used to create files required for constructing dependency
tree used in semantic relatedness.

constituency_tree() is used to created files required for constructing constituency
tree used in sentiment classification. Here it is not used as the tree is directly
constructed from a file in setiment tree bank.
**************************************************************************************************
"""

def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, 'dparents.txt')
    relpath =  os.path.join(dirpath, 'rels.txt')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)


def constituency_parse(filepath, cp='', tokenize=True):
    print('\n Constituency parsing ' + filepath)
    print('This takes a lot of time...')
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
           % (cp, tokpath, parentpath, tokenize_flag, filepath))
    os.system(cmd)
