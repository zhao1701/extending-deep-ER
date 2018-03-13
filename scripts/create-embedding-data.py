
import numpy as np
import pandas as pd
import pickle as pkl
import argparse as ap

from gensim.models import KeyedVectors
from collections import defaultdict

# parse command-line arguments
parser = ap.ArgumentParser()
parser.add_argument('embedding',
                    help = 'choose embedding type (word2vec or glove)')
parser.add_argument('source', help = 'file path of downloaded embedding data')
parser.add_argument('-d', '--destination',
                    help = 'directory path to generate new embedding files',
                    default = '../data/embeddings/')

args = parser.parse_args()
embedding = args.embedding
source = args.source
destination = args.destination

# create and save gensim embedding model
print('creating gensim models')
if embedding == 'glove':
    model_name = 'glove'
    model = KeyedVectors.load_word2vec_format(source, binary=False)
    
elif embedding == 'word2vec':
    model_name = 'word2vec'
    model = KeyedVectors.load_word2vec_format(source, binary=True)
    
elif embedding == 'debug':
    print(embedding)
    print(source)
    print(destination)
    quit()

else:
    raise 'Not a valid embedding type'

model.save(destination + '{}-300.gensim'.format(model_name))

# create and save numpy embedding matrix with initial row of zeros
print('creating embedding matrix')
embedding_matrix = model.vectors
embedding_matrix = np.vstack([np.zeros(300), embedding_matrix])
np.save(file='../data/embeddings/{}-300.matrix'.format(model_name),
        arr=embedding_matrix)

# create and save two maps of corpus vocabulary
print('creating maps')
vocab = ['<unk>'] + list(model.vocab.keys())
word2idx = defaultdict(int, zip(vocab, range(len(vocab))))
idx2word = dict(zip(range(len(vocab)), vocab))

# manually encode NaN's as unknown
for nan in ['NaN', 'NAN', 'nan', 'Nan']:
    word2idx[nan] = 0

map = dict()
map['word2idx'] = word2idx
map['idx2word'] = idx2word

with open('../data/embeddings/{}-300.map'.format(model_name), 'wb') as f:
    pkl.dump(map, f)