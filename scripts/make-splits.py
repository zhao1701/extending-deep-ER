
import pandas as pd
import numpy as np
import argparse as ap
import os

from collections import defaultdict

parser = ap.ArgumentParser()
parser.add_argument('source_dir',
                    help='directory containing dataset and match files to split')
parser.add_argument('dest_dir',
                    help='directory to save split dataset csvs')
parser.add_argument('--set1', '-s1', default='set1.csv',
                    help='filename of first dataset csv')
parser.add_argument('--set2', '-s2', default='set2.csv',
                    help='filename of second dataset csv')
parser.add_argument('--matches', '-m', default='matches.csv',
                    help='filename of positives matches csv')
parser.add_argument('--neg_pos_ratio', '-npr', default = 9, type=float,
                    help='ratio of non-matching pairs to matching pairs')
parser.add_argument('--val_prop', '-vp', default = 0.1, type=float,
                    help='proportion of data to allocate to validation set')
parser.add_argument('--test_prop', '-tp', default = 0.1, type=float,
                    help='proportion of data to allocate to test set')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='print statistics')

# parse arguments
args = parser.parse_args()
source_dir = args.source_dir
set1 = args.set1
set2 = args.set2
matches = args.matches

destination_path = args.dest_dir

neg_pos_ratio = args.neg_pos_ratio
val_prop = args.val_prop
test_prop = args.test_prop

verbose = args.verbose

df1 = pd.read_csv(os.path.join(source_dir, set1), encoding = "latin1")
df1['id1'] = df1['id1'].astype(str)

df2 = pd.read_csv(os.path.join(source_dir, set2), encoding = "latin1")
df2['id2'] = df2['id2'].astype(str)

df_pos = pd.read_csv(os.path.join(source_dir, matches), encoding = "latin1")
df_pos['id1'] = df_pos['id1'].astype(str)
df_pos['id2'] = df_pos['id2'].astype(str)

# calculate number of matches available and...
# number of non-matches that need to be sampled
n_positives = len(df_pos)
n_negatives = int(neg_pos_ratio * n_positives)
if verbose:
    print('{} matched pairs detected. Creating {} non-matched pairs.'.\
          format(n_positives, n_negatives))
    
# extract id columns from respective datasets
id1 = df1['id1'].astype(str)
id2 = df2['id2'].astype(str)

# create a mapping from id1 to a list of matches in id2.
# when creating non-matches, we can consult dictmap to ensure non-matches...
# are not accidentally constructed from matched pairs
pos_map = defaultdict(list)
for row in df_pos.iterrows():
    id1_val = row[1]['id1']
    id2_val = row[1]['id2']
    pos_map[id1_val].append(id2_val)

def drop_positives(df_negs, pos_map):
    "drops positive matches from dataframe of non-matches"
    df_negs = df_negs.copy()
    for index, row in df_negs.iterrows():
        if row['id2'] in pos_map[row['id1']]:
            df_negs.drop(index, inplace=True)
    return df_negs

# create a set of [n_negatives] non-matches by sampling from [id1] and [id2]...
# with replacement. because dropping duplicates and matches results in a...
# lower count, we oversample then filter. if oversampling not sufficient...
# repeat process with progressively larger oversampling multipliers until
# [> n_negatives] final non-matches achieved.

oversample_base = 1.2
oversample_exp = 0
df_negs = pd.DataFrame(columns = ['id1', 'id2'])

while len(df_negs) < n_negatives:
    oversample_exp += 1
    oversample_multiplier = oversample_base ** oversample_exp
    n_negatives_os = int(oversample_multiplier * n_negatives)
    
    id1_negs = id1.sample(n_negatives_os, replace=True).reset_index(drop=True)
    id2_negs = id2.sample(n_negatives_os, replace=True).reset_index(drop=True)
    
    df_negs = pd.concat([id1_negs, id2_negs], axis = 'columns')
    df_negs = df_negs.drop_duplicates().reset_index(drop=True)
    df_negs = drop_positives(df_negs, pos_map)
    df_negs = df_negs.iloc[:n_negatives,:]
    
# ensure all generated non-match pairs are not matches
for index, row in df_negs.iterrows():
    assert((row['id2'] in pos_map[row['id1']]) == False)

# add target column to both positive and negative sets
df_negs['match'] = 0
df_pos['match'] = 1

# vertically stack dataframes and shuffle
df = pd.concat([df_negs, df_pos], axis = 'rows')
df = df.sample(len(df), replace=False)

# ensure all pairs in df are labeled correctly
for index, row in df.iterrows():
    assert((row['id2'] in pos_map[row['id1']]) == row['match'])

# calculate indices on which to split dataset into test and validation sets
test_idx = np.round(len(df) * test_prop).astype(int)
val_idx = np.round(len(df) * (test_prop + val_prop)).astype(int)

df_test = df.iloc[:test_idx,:]
df_val = df.iloc[test_idx:val_idx,:]
df_train = df.iloc[val_idx:,:]

# merge in relevant attributes from each dataset to id's in train, val, ...
# and test set

df_train_1 = pd.merge(df_train, df1, how='left',  on=['id1'])
df_train_1 = df_train_1.drop(['id2', 'match'], axis='columns')
df_train_2 = pd.merge(df_train, df2, how='left',  on=['id2'])
df_train_2 = df_train_2.drop(['id1', 'match'], axis='columns')
df_train_y = df_train['match']

df_val_1 = pd.merge(df_val, df1, how='left',  on=['id1'])
df_val_1 = df_val_1.drop(['id2', 'match'], axis='columns')
df_val_2 = pd.merge(df_val, df2, how='left',  on=['id2'])
df_val_2 = df_val_2.drop(['id1', 'match'], axis='columns')
df_val_y = df_val['match']

df_test_1 = pd.merge(df_test, df1, how='left',  on=['id1'])
df_test_1 = df_test_1.drop(['id2', 'match'], axis='columns')
df_test_2 = pd.merge(df_test, df2, how='left',  on=['id2'])
df_test_2 = df_test_2.drop(['id1', 'match'], axis='columns')
df_test_y = df_test['match']

# ensure all id's match
assert(np.all(df_train_1['id1'].values == df_train['id1'].values))
assert(np.all(df_train_2['id2'].values == df_train['id2'].values))
assert(np.all(df_val_1['id1'].values == df_val['id1'].values))
assert(np.all(df_val_2['id2'].values == df_val['id2'].values))
assert(np.all(df_test_1['id1'].values == df_test['id1'].values))
assert(np.all(df_test_2['id2'].values == df_test['id2'].values))

if verbose:
    print('Training set contains {} instances'.format(len(df_train_y)))
    print('Validation set contains {} instances'.format(len(df_val_y)))
    print('Test set contains {} instances'.format(len(df_test_y)))
    
if not os.path.isdir(destination_path):
    os.mkdir(destination_path)
    if verbose:
        print('Creating destination directory.')
        
# convert 'y' Series to dataframes to avoid header import mismatches
df_train_y = pd.DataFrame(df_train_y)
df_val_y = pd.DataFrame(df_val_y)
df_test_y = pd.DataFrame(df_test_y)

# save newly split dataframes in specified destination
df_train_1.to_csv(os.path.join(destination_path, 'train_1.csv'), index=False)
df_train_2.to_csv(os.path.join(destination_path, 'train_2.csv'), index=False)
df_train_y.to_csv(os.path.join(destination_path, 'train_y.csv'), index=False)

df_val_1.to_csv(os.path.join(destination_path, 'val_1.csv'), index=False)
df_val_2.to_csv(os.path.join(destination_path, 'val_2.csv'), index=False)
df_val_y.to_csv(os.path.join(destination_path, 'val_y.csv'), index=False)

df_test_1.to_csv(os.path.join(destination_path, 'test_1.csv'), index=False)
df_test_2.to_csv(os.path.join(destination_path, 'test_2.csv'), index=False)
df_test_y.to_csv(os.path.join(destination_path, 'test_y.csv'), index=False)