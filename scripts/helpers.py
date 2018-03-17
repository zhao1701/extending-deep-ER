import pandas as pd
import numpy as np
import pickle as pkl
import os

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

def df_idx_to_words(x, mapping_file = '../data/embeddings/glove-300.map'):
    
    with open(mapping_file, 'rb') as f:
        map = pkl.load(f)
        
    def idx_list_to_words(x):
        string = ''
        for idx in x:
            string = string + ' ' + map['idx2word'][idx]
        return string
    
    df = x.copy()
    idx_mask = x.apply(is_str_list)
    df_idx = df.loc[:, idx_mask]
    df.loc[:, idx_mask] = df.loc[:, idx_mask].applymap(idx_list_to_words)
    return df

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