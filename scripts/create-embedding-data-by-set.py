
import os
import shutil
import re
import string
import html

import numpy as np
import pandas as pd
import pickle as pkl
import argparse as ap

from gensim.models import KeyedVectors
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

# parse command-line arguments
parser = ap.ArgumentParser()
parser.add_argument('embedding_type',
                    help = 'choose embedding type (word2vec or glove)')
parser.add_argument('embedding_file',
                    help = 'file path of downloaded embedding data')
parser.add_argument('data_dir',
                    help = 'directory containing dataset and match files')
parser.add_argument('dest_dir',
                    help = 'directory path to generate new files')
parser.add_argument('--columns', '-c', nargs='+', required=True,
                    help = 'names of columns to be converted')
parser.add_argument('--drop', '-d', nargs='+', required=False,
                    help = 'names of columns to be dropped')
parser.add_argument('--set1', '-s1', default='set1.csv',
                    help='filename of first dataset csv')
parser.add_argument('--set2', '-s2', default='set2.csv',
                    help='filename of second dataset csv')
parser.add_argument('--matches', '-m', default='matches.csv',
                    help='filename of positives matches csv')
parser.add_argument('--sklearn', '-s', action='store_true',
                    help='whether to use sklearn\'s CountVectorizer')
parser.add_argument('--max_df', default=0.1, type=float,
                    help='proportion of documents above which token will be excluded')

args = parser.parse_args()
embedding_type = args.embedding_type
embedding_file = args.embedding_file
dest_dir = args.dest_dir
data_dir = args.data_dir
set1 = args.set1
set2 = args.set2
matches = args.matches
columns = args.columns
drop = args.drop
use_sklearn = args.sklearn
max_df = args.max_df

print('Loading Gensim model...')
if embedding_type in ['glove', 'word2vec']:
    model = KeyedVectors.load(embedding_file)      
elif embedding_type == 'debug':
    print(embedding_type)
    print(embedding_file)
    print(dest_dir)
    quit()
else:
    raise 'Not a valid embedding type'
gensim_vocab = defaultdict(int, model.vocab)

df1 = pd.read_csv(os.path.join(data_dir, set1), encoding = "latin1")
df2 = pd.read_csv(os.path.join(data_dir, set2), encoding = "latin1")

# check input data meets requirements
df1_column_check = list(df1.columns)
df2_column_check = list(df2.columns)

print('Check id column names are valid: ', end='')
assert('id1' in df1_column_check)
assert('id2' in df2_column_check)
print('passed')

df1_column_check.remove('id1')
df2_column_check.remove('id2')

print('Check datasets have same column names: ', end='')
assert(df1_column_check == df2_column_check)
print('passed')

print('Check listed columns are valid: ', end='')
for column in columns:
    assert(column in df1_column_check)
if drop:
    for column in drop:
        assert(column in df1_column_check)
print('passed')

print('Columns to convert:')
for column in columns:
    print('\t' + column)

if drop:
    print('Columns to drop:')
    for column in drop:
        print('\t' + column)
        df1 = df1.drop(column, axis='columns')
        df2 = df2.drop(column, axis='columns')

if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

# no need to do anything to matches.csv, so just copy it to destination
matches_source_path = os.path.join(data_dir, matches)
matches_dest_path = os.path.join(dest_dir, matches)
shutil.copyfile(matches_source_path, matches_dest_path)

# pre-process all text data into tokens compatible with Glove/Word2Vec
def clean_text(x):
    "formats a single string"
    if not isinstance(x, str):
        return 'NaN'
    
    # separate possessives with spaces
    x = x.replace('\'s', ' \'s')
    
    # convert html escape characters to regular characters
    x = html.unescape(x)
    
    # separate punctuations with spaces
    def pad(x):
        match = re.findall(r'.', x[0])[0]
        match_clean = ' ' + match + ' '
        return match_clean
    rx = r'\(|\)|/|!|#|\$|%|&|\\|\*|\+|,|:|;|<|=|>|\?|@|\[|\]|\^|_|{|}|\||'
    rx += r'`|~'
    x = re.sub(rx, pad, x)
    
    # remove decimal parts of version numbers
    def v_int(x):
        return re.sub('\.\d+','',x[0])
    x = re.sub(r'v\d+\.\d+', v_int, x)
    
    return x

print('Cleaning text. ', end = '')
df1.loc[:, columns] = df1.loc[:, columns].applymap(clean_text)
df2.loc[:, columns] = df2.loc[:, columns].applymap(clean_text)

# for any tokens not in model vocabulary, try a few capitalization variants
fixed = list()
def check_tokens(x):
    global fixed
    x = x.split()
    new_string = ''
    for token_orig in x:
        token = token_orig
        if not bool(gensim_vocab[token]):
            token = token.lower()
            if bool(gensim_vocab[token]):
                fixed.append(token_orig)
        if not bool(gensim_vocab[token]):
            token = string.capwords(token)
            if bool(gensim_vocab[token]):
                fixed.append(token_orig)
        if not bool(gensim_vocab[token]):
            token = token.upper()
            if bool(gensim_vocab[token]):
                fixed.append(token_orig)
        new_string = new_string + ' ' + token
    return new_string
df1.loc[:, columns] = df1.loc[:, columns].applymap(check_tokens)
df2.loc[:, columns] = df2.loc[:, columns].applymap(check_tokens)
print('Fixed {} unique tokens'.format(pd.Series(fixed).nunique()))
        
# map each token to an index and convert text fields accordingly

print('Creating map file. ', end='')
# collapse all text columns in both datasets to a single list of strings
corpus = list()
for df in [df1, df2]:
    for column in columns:
        corpus.extend(list(df[column]))

# map each token to a unique non-zero index
word2idx = defaultdict(int)
idx2word = defaultdict(str)
if not use_sklearn:
    i = 1
    # missing = ['nan', 'NAN', 'Nan', 'NaN']
    for instance in corpus:
        for token in instance.split():
            if not word2idx[token]:
                word2idx[token] = i
                i += 1

    # create a reverse mapping from index to token
    for key, value in word2idx.items():
        idx2word[value] = key
else:
    cv = CountVectorizer(max_df=max_df)
    cv.fit(corpus)
    sk_dict = cv.vocabulary_
    # offset indices assigned by sklearn so 0 index is free
    for word, index in sk_dict.items():
        word2idx[word] = index + 1
        idx2word[index+1] = word
    
print('{} unique tokens detected.'.format(len(word2idx)))
    
print('Building embedding matrix. ', end='')
# an extra row of zeros at top of matrix is needed for Keras zero padding
embedding_matrix = np.zeros([len(word2idx) + 1, 300])
n_unknowns = 0
for word, index in word2idx.items():
    # if word has no vector embedding, leave corresponding row to be an
    # ...attenuated random Gaussian
    if bool(gensim_vocab[word]):
        embedding_vector = model.get_vector(word)      
    else:
        embedding_vector = np.random.randn(300) / 300
        n_unknowns += 1
    embedding_matrix[index, :] = embedding_vector
print('{} unknown tokens assigned random Gaussian.'.format(n_unknowns))
    
print('Converting text data to index vectors.')
def record2idx(x):
    if not use_sklearn:
        x = x.split()
    else:
        x = x.lower().split()
    return [word2idx[word] for word in x]

df1.loc[:, columns] = df1.loc[:, columns].applymap(record2idx)
df2.loc[:, columns] = df2.loc[:, columns].applymap(record2idx)

# save files
df1.to_csv(os.path.join(dest_dir, set1), index=False)
df2.to_csv(os.path.join(dest_dir, set2), index=False)
np.save(arr=embedding_matrix,
        file=os.path.join(dest_dir, '{}-300.matrix'.format(embedding_type)))

# save both word2idx and idx2word mappings into a double dictionary
map = dict(word2idx = word2idx, idx2word = idx2word)
with open(os.path.join(dest_dir, embedding_type + '-300.map'), 'wb') as f:
    pkl.dump(map, f)