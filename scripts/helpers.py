import pandas as pd
import numpy as np
import os
import re

# DATA HANDLING
def is_str_list(x):
    """
    given a pd.Series of strings, return True if all elements
    begin and end with square brackets
    """
    return np.all(x.astype(str).str.startswith('[') & \
                  x.astype(str).str.endswith(']'))

def str_to_list(x):
    "convert a string reprentation of list to actual list"
    x = x[1:-1]
    x = x.split(',')
    return [int(i) for i in x]

def load_data(data_dir, filenames=['test_1', 'test_2', 'test_y',
                                   'train_1', 'train_2', 'train_y',
                                   'val_1', 'val_2', 'val_y']):
    """
    returns a dictionary of test, train, and validation datasets with their
    respective sources and targets. filenames serve as keys.
    """
    data = dict()
    for filename in filenames:
        df = pd.read_csv(os.path.join(data_dir, filename+'.csv')) 
        str_list_mask = df.apply(is_str_list, axis='rows')
        df.loc[:, str_list_mask] = df.loc[:, str_list_mask].applymap(str_to_list)
        data[filename] = df
    return data

def str_to_list_df(x):
    df = x.copy()
    mask = df.apply(is_str_list, axis='rows')
    df.loc[:, mask] = df.loc[:, mask].applymap(str_to_list)
    return df

def str_to_num(x):
    return float(re.sub('[^0-9|^\.]', '', x))

def examine_data(set1, set2, columns, bool_mask, mapping):
    
    df1 = set1.copy()
    df2 = set2.copy()
    
    def idx_to_word(x):
        string = ''
        for idx in x:
            string += ' ' + mapping['idx2word'][idx]
        return string
    
    df1.loc[:, columns] = df1.loc[:, columns].applymap(idx_to_word)
    df2.loc[:, columns] = df2.loc[:, columns].applymap(idx_to_word)
    
    both = pd.concat([df1, df2], axis=1)
    both = both.loc[bool_mask, :]
    return both

# HYPEROPT VISUALIZATIONS

def hyperopt_val_diagnostic(val_name, trials):
        
    ts = [trial['tid'] for trial in trials.trials]
    results = [trial['result']['loss'] for trial in trials.trials]
    
    fig, axes = plt.subplots(1, 3, figsize = (16,4))
    axes[0].scatter(ts, vals)
    axes[0].set(xlabel='iteration', ylabel=val_name)
    axes[1].hist(np.array(vals).squeeze())
    axes[1].set(xlabel=val_name, ylabel='frequency')
    axes[2].scatter(vals, results)
    axes[2].set(xlabel=val_name, ylabel='loss')
    plt.tight_layout()
    
def visualize_hyperparameters(trials):
    for val in trials.trials[0]['misc']['vals'].keys():
        hyperopt_val_diagnostic(val, trials)
        
# HELPERS FOR MODEL GENERATION

def get_document_frequencies(raw_data_dir, mapping, set1='set1', set2='set2'):
    
    # read csv data from directory as pd.DataFrame
    set1 = pd.read_csv(os.path.join(raw_data_dir, set1 + '.csv'), encoding='latin1')
    set2 = pd.read_csv(os.path.join(raw_data_dir, set2 + '.csv'), encoding='latin1')
    
    # select only columns whose values are lists embedded as strings
    mask1 = set1.apply(is_str_list, axis='rows')
    mask2 = set2.apply(is_str_list, axis='rows')
    
    # convert strings back into lists
    set1 = set1.loc[:, mask1].applymap(str_to_list)
    set2 = set2.loc[:, mask2].applymap(str_to_list)
    
    
    # concatenate columns so all relevant attributes become a single list
    def concat_columns(x):
        idx_list = list()
        for lst in x.values:
            idx_list += lst
        return idx_list
    
    set1 = set1.apply(concat_columns, axis='columns')
    set2 = set2.apply(concat_columns, axis='columns')
    
    # +1 because default value of DefaultDict not counted
    doc_freqs_1 = np.zeros(len(mapping['idx2word'])+1)
    doc_freqs_2 = np.zeros(len(mapping['idx2word'])+1)
    
    for index, item in set1.iteritems():
        uniq_indices = set(item)
        for idx in uniq_indices:
            doc_freqs_1[idx] += 1
    
    for index, item in set2.iteritems():
        uniq_indices = set(item)
        for idx in uniq_indices:
            doc_freqs_2[idx] += 1
    
    return doc_freqs_1, doc_freqs_2