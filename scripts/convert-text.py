import pandas as pd
import numpy as np
import pickle as pkl
import argparse as ap

import os
import re
import string
import html

from collections import defaultdict

parser = ap.ArgumentParser()
parser.add_argument('source_dir',
                    help='directory containing dataset and match files to split')
parser.add_argument('dest_dir',
                    help='directory to save split dataset csvs')
parser.add_argument('mapping_file',
                    help='double dictionary containing maps to-from words\
                          and vocabulary indices')
parser.add_argument('--set1', '-s1', default='set1.csv',
                    help='filename of first dataset csv')
parser.add_argument('--set2', '-s2', default='set2.csv',
                    help='filename of second dataset csv')
parser.add_argument('--matches', '-m', default='matches.csv',
                    help='filename of positives matches csv')
parser.add_argument('--indices', '-i', nargs='+', type=int,
                    help='indices of columns to be converted (starting from 0)')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='print statistics')

# parse command line arguments
args = parser.parse_args()
source_dir = args.source_dir
dest_dir = args.dest_dir
mapping_file = args.mapping_file
column_idxs = args.indices

verbose = args.verbose

set1 = args.set1
set2 = args.set2
matches = args.matches

if verbose:
    print('Loading datasets and maps.')
# load data
# df_pos is loaded so that it can be copied to destination directory
df1 = pd.read_csv(os.path.join(source_dir, set1), encoding = "latin1")
df2 = pd.read_csv(os.path.join(source_dir, set2), encoding = "latin1")
df_pos = pd.read_csv(os.path.join(source_dir, matches), encoding = "latin1")

# make column names the same
assert(df1.columns[0] == 'id1')
assert(df2.columns[0] == 'id2')
df2.columns = [df2.columns[0]] + list(df1.columns[1:])

# load double dictionary
with open(mapping_file, 'rb') as f:
    map = pkl.load(f)

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

if verbose:
    print('Cleaning the following columns from set1:')
    for column in df1.columns[column_idxs]:
        print(column, end=' ')
    print()
    print('Cleaning the following columns from set2:')
    for column in df2.columns[column_idxs]:
        print(column, end=' ')
    print()

df1.iloc[:, column_idxs] = df1.iloc[:, column_idxs].applymap(clean_text)
df2.iloc[:, column_idxs] = df2.iloc[:, column_idxs].applymap(clean_text)

def record2idx(x):
    x = x.split()
    for i, token in enumerate(x):
        idx = map['word2idx'][token]
        if idx == 0:
            idx = map['word2idx'][token.lower()]
        if idx == 0:
            idx = map['word2idx'][string.capwords(token)]
        if idx == 0:
            idx = map['word2idx'][token.upper()]
        x[i] = idx
    return x

if verbose:
    print('Converting tokens to indices.')
df1.iloc[:, column_idxs] = df1.iloc[:, column_idxs].applymap(record2idx)
df2.iloc[:, column_idxs] = df2.iloc[:, column_idxs].applymap(record2idx)

if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)
    if verbose:
        print('Creating destination directory')
    
df1.to_csv(os.path.join(dest_dir, set1), index=False)
df2.to_csv(os.path.join(dest_dir, set2), index=False)
df_pos.to_csv(os.path.join(dest_dir, matches), index=False)