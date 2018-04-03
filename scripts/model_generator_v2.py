
import sys
import os
import re
sys.path.append('../scripts')

import numpy as np
import pandas as pd
import helpers as hp
import pickle as pkl
import itertools as it
import tensorflow as tf

from keras.layers import Input, Dense, LSTM, Embedding, Lambda, Dot, Concatenate, \
                         Bidirectional, BatchNormalization, Dropout
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adadelta
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras import backend as K
from collections import OrderedDict, defaultdict

def deep_er_model_generator_v2(data_dict,
                            embedding_file,
                            padding_limit = 100,
                            mask_zero = True,
                            embedding_trainable = False,
                            text_columns = list(), 
                            numeric_columns = list(),
                            make_isna = True,
                            text_nan_idx = None,
                            num_nan_val = None,
                            text_compositions = ['average'],
                            text_sim_metrics = ['cosine'],
                            numeric_sim_metrics = ['unscaled_inverse_lp'],
                            dense_nodes = [10],
                            lstm_args = dict(units=50),
                            document_frequencies = None,
                            idf_smoothing = 2,
                            batch_norm = False,
                            dropout = 0):
    """
    Takes a dictionary of paired split DataFrames and returns a DeepER 
    model with data formatted for said model.
    
    Parameters
    ----------
    data_dict : dict
        A dictionary of dataframes (pd.DataFrame) stored with the following
        keys: train_1, val_1, test_1, train_2, val_2, test_2
    embedding_file : str
        The location and name of numpy matrix containing word vector
        embeddings.
    padding_limit : int, optional
        The maximum length of any text sequence. For any text attribute whose
        max length is below padding_limit, the max length will be used.
        Otherwise, padding_limit will be used to both pad and truncuate
        text sequences for that attribute.
    mask_zero : bool, optional
        Whether to ignore text sequence indices with value of 0. Useful for
        LSTM's and variable length inputs.
    embedding_trainable: bool, optional
        Whether to allow the embedding layer to be fine tuned.
    text_columns : list of strings, optional
        A list of names of text-based attributes
    numeric_columns : list of strings, optional
        A list of names of numeric attributes
    make_isna: bool, optional
        Whether to create new attributes indicating the presence of null values
        for each original attribute.
    text_nan_idx : int, optional
        The index corresponding to NaN values in text-based attributes.
    num_nan_val : int, optional
        The value corresponding to NaN values in numeric attributes.
    text_compositions : list of strings, optional
        List of composition methods to be applied to embedded text attributes.
        Valid options are :
            - average : a simple average of all embedded vectors
            - idf : an average of all embedded vectors weighted by normalized
                    inverse document frequency
            - bi_lstm
    text_sim_metrics : list of strings, optional
        List of similarity metrics to be computed for each text-based attribute.
        Valid options are :
            - cosine
            - inverse_l1 : e^-[l1_distance]
            - inverse_l2 : e^-[l2_distance]
    numeric_sim_metrics : list of strings, optional
        List of similarity metrics to be computed for each numeric attribute.
        Valid options are :
            - scaled_inverse_lp : e^[-2(abs_diff)/sum]
            - unscaled_inverse_lp : e^[-abs_diff]
            - min_max_ratio : min / max
    dense_nodes : list of ints, optional
        Specifies topology of hidden dense layers
    lstm_args = dict, optional
        Keyword arguments for LSTM layer
    document_frequencies = tuple of length 2, optional
        Tuple of two lists of document frequencies, left side then right
    idf_smoothing : int, optional
        un-normalized idf = 1 / df ^ (1 / idf_smoothing)
        Higher values means that high document frequency words are penalized
        less.
    """
    
    ### DATA PROCESSING ###
    # initialize an empty dictionary for storing all data
    # dictionary structure will be data[split][side][column]
    sides = ['left', 'right']
    splits = ['train', 'val', 'test']
    data = dict()
    for split in splits:
        data[split] = dict()
        for side in sides:
            data[split][side] = dict()
            
    columns = text_columns + numeric_columns
    
    # separate each feature into its own dictionary entry
    for column in columns:
        data['train']['left'][column] = data_dict['train_1'][column]
        data['train']['right'][column] = data_dict['train_2'][column]

        data['val']['left'][column] = data_dict['val_1'][column]
        data['val']['right'][column] = data_dict['val_2'][column]

        data['test']['left'][column] = data_dict['test_1'][column]
        data['test']['right'][column] = data_dict['test_2'][column]
    
    # if enabled, create a binary column for each feature indicating whether
    # it contains a missing value. for text data, this will be a list with
    # a single index representing the 'NaN' token. for numeric data, this will
    # likely be a 0.
    if make_isna:
        for split, side, column in it.product(splits, sides, text_columns):
            isna = data[split][side][column].apply(lambda x: x == [text_nan_idx])
            isna = isna.values.astype(np.float32).reshape(-1, 1)
            isna_column = column + '_isna'
            data[split][side][isna_column] = isna
        for split, side, column in it.product(splits, sides, numeric_columns):
            isna = data[split][side][column].apply(lambda x: x == num_nan_val)
            isna_column = column + '_isna'
            isna = isna.values.astype(np.float32).reshape(-1, 1)
            data[split][side][isna_column] = isna
    
    # pad each text column according to the length of its longest entry in
    # both datasets
    maxlen = dict()
    for column in text_columns:
        maxlen_left = data['train']['left'][column].apply(lambda x: len(x)).max()
        maxlen_right = data['train']['right'][column].apply(lambda x: len(x)).max()
        maxlength = min(padding_limit, max(maxlen_left, maxlen_right))
        for split, side in it.product(splits, sides):
            data[split][side][column] = pad_sequences(data[split][side][column],
                                                      maxlen=maxlength,
                                                      padding='post',
                                                      truncating='post')
        maxlen[column] = maxlength
    
    # convert all numeric features to float and reshape to be 2-dimensional
    for split, side, column in it.product(splits, sides, numeric_columns):
        feature = data[split][side][column]
        feature = feature.values.astype(np.float32).reshape(-1,1)
        data[split][side][column] = feature
            
    # format X values for each split as a list of 2-dimensional arrays
    packaged_data = OrderedDict()
    for split in splits:
        packaged_data[split] = list()
        for side, column in it.product(sides, columns):
            packaged_data[split].append(data[split][side][column])
        if make_isna:
            for side, column in it.product(sides, columns):
                packaged_data[split].append(data[split][side][column + '_isna'])
    
    # convert y-values
    y_train = to_categorical(data_dict['train_y'])
    y_val = to_categorical(data_dict['val_y'])
    y_test = to_categorical(data_dict['test_y'])
    
    ### MODEL BUILDING ###
    
    # each attribute of each side is its own input tensor
    # text input tensors for both sides are created before numeric input tensors
    input_tensors = dict(left=dict(), right=dict())
    for side, column in it.product(sides, text_columns):
        input_tensors[side][column] = Input(shape=(maxlen[column],))
        
    for side, column in it.product(sides, numeric_columns):
        input_tensors[side][column] = Input(shape=(1,))
    
    # create a single embedding layer for text input tensors
    embedding_matrix = np.load(embedding_file)
    embedding_layers = dict()
    for composition in text_compositions:
        embedding_layers[composition] = dict()
        for similarity in text_sim_metrics:
            embedding_layers[composition][similarity] = Embedding(embedding_matrix.shape[0],
                                                        embedding_matrix.shape[1],
                                                        weights=[embedding_matrix],
                                                        trainable=embedding_trainable,
                                                        mask_zero=mask_zero)
            
    
    # use embedding_layer ot convert text input tensors to embedded tensors
    # and store in dictionary.
    # an embedding tensor will have shape n_words x n_embedding_dimensions
    embedded_tensors = dict()
    for composition in text_compositions:
        embedded_tensors[composition] = dict()
        for similarity in text_sim_metrics:
            embedded_tensors[composition][similarity] = dict(left=dict(), right=dict())
            for side, column in it.product(sides, text_columns):
                embedded_tensors[composition][similarity][side][column] = \
                embedding_layers[composition][similarity](input_tensors[side][column])
    
    # initialize dictionary for storing a composition tensor for each embedding tensor
    composed_tensors = dict()
    for composition in text_compositions:
        composed_tensors[composition] = dict()
        for similarity in text_sim_metrics:
            composed_tensors[composition][similarity] = dict()
            for side in sides:
                composed_tensors[composition][similarity][side] = dict()
    
    # if enabled, reduce each embedding tensor to a quasi-1-dimensional tensor
    # with shape 1 x n_embedding_dimensions by averaging all embeddings
    if 'average' in text_compositions:
        averaging_layer = Lambda(lambda x: K.mean(x, axis=1), output_shape=(maxlen[column],))
        for similarity in text_sim_metrics:
            for side, column in it.product(sides, text_columns):
                composed_tensors['average'][similarity][side][column] = \
                averaging_layer(embedded_tensors['average'][similarity][side][column])
    
    # if enabled, reduce each embedding tensor to a quasi-1-dimensional tensor
    # with shape 1 x n_embedding_dimensions by taking a weighted average of all
    # embeddings. 
    if 'idf' in text_compositions:
        # store document frequency constants for each side
        dfs_constant = dict()
        dfs_constant['left'] = K.constant(document_frequencies[0])
        dfs_constant['right'] = K.constant(document_frequencies[1])
        
        # a selection layer uses an input tensor as indices to select
        # document frequencies from dfs_constant
        dfs_selection_layer = dict()
        
        # a conversion layer converts a tensor of selected document frequencies
        # to a tensor of inverse document frequencies. the larger the DF,
        # the smaller the inverse, the smallness of which is controlled by
        # idf_smoothing
        idf_conversion_layer = Lambda(lambda x: 1 / (K.pow(x, 1/idf_smoothing)))
        
        # document frequencies of 0 will result in IDF's of inf. these should
        # be converted back to 0's.
        idf_fix_layer = Lambda(lambda x: tf.where(tf.is_inf(x), tf.zeros_like(x), x))
        
        # for each IDF tensor, scale its values so they sum to 1
        idf_normalization_layer = Lambda(lambda x: x / K.expand_dims(K.sum(x, axis=1), axis=1))
        
        # take dot product between embedding tensor vectors and IDF weights
        dot_layer = Dot(axes=1)
        
        for similarity in text_sim_metrics:
            for side in sides:
                dfs_selection_layer[side] = Lambda(lambda x: K.gather(dfs_constant[side], K.cast(x, tf.int32)))
                for column in text_columns:                
                    dfs_tensor = dfs_selection_layer[side](input_tensors[side][column])
                    idfs_tensor = idf_conversion_layer(dfs_tensor)
                    idfs_tensor_fixed = idf_fix_layer(idfs_tensor)
                    idfs_tensor_normalized = idf_normalization_layer(idfs_tensor_fixed)
                    composed_tensors['idf'][similarity][side][column] = \
                    dot_layer([embedded_tensors['idf'][similarity][side][column], idfs_tensor_normalized])
                
    # if enabled, compose embedding tensor using LSTM        
    if 'lstm' in text_compositions:
        for similarity in text_sim_metrics:
            for side, column in it.product(sides, text_columns):
                lstm_layer = LSTM(**lstm_args)
                composed_tensors['lstm'][similarity][side][column] = \
                lstm_layer(embedded_tensors['lstm'][similarity][side][column])
    
    # if enambled, compose embedding tensor using bi-directional LSTM
    if 'bi_lstm' in text_compositions:
        for similarity in text_sim_metrics:
            for side, column in it.product(sides, text_columns):
                lstm_layer = Bidirectional(LSTM(**lstm_args), merge_mode='concat')
                composed_tensors['bi_lstm'][similarity][side][column] = \
                lstm_layer(embedded_tensors['bi_lstm'][similarity][side][column])
    
    # maintain list of text-based similarities to calculate
    similarity_layers = dict()
    if 'cosine' in text_sim_metrics:
        similarity_layer = Dot(axes=1, normalize=True)
        similarity_layers['cosine'] = similarity_layer
    if 'inverse_l1' in text_sim_metrics:
        similarity_layer = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))
        similarity_layers['inverse_l1'] = similarity_layer
    if 'inverse_l2' in text_sim_metrics:
        similarity_layer = Lambda(lambda x: \
                                  K.exp(-K.sqrt(K.sum(K.pow(x[0]-x[1], 2), axis=1, keepdims=True))))
        similarity_layers['inverse_l2'] = similarity_layer
    
    # for each attribute, calculate similarities between left and ride sides
    similarity_tensors = list()
    for composition, column, similarity in \
        it.product(text_compositions, text_columns, text_sim_metrics):     
        similarity_tensor = similarity_layers[similarity]([composed_tensors[composition][similarity]['left'][column],
                                                           composed_tensors[composition][similarity]['right'][column]])
        similarity_tensors.append(similarity_tensor)
    
    # reset similarity layer to empty so only numeric-based similarities are used
    similarity_layers = list()
    if 'scaled_inverse_lp' in numeric_sim_metrics:
        similarity_layer = Lambda(lambda x: K.exp(-2 * K.abs(x[0]-x[1]) / (x[0] + x[1])))
        similarity_layers.append(similarity_layer)
    if 'unscaled_inverse_lp' in numeric_sim_metrics:
        similarity_layer = Lambda(lambda x: K.exp(-K.abs(x[0]-x[1])))
        similarity_layers.append(similarity_layer)
        
    for column, similarity_layer in it.product(numeric_columns, similarity_layers):
        similarity_tensor = similarity_layer([input_tensors['left'][column],
                                              input_tensors['right'][column]])
        similarity_tensors.append(similarity_tensor)
    if 'min_max_ratio' in numeric_sim_metrics:
        for column in numeric_columns:
            num_concat = Concatenate(axis=-1)([input_tensors['left'][column], input_tensors['right'][column]])
            similarity_layer = Lambda(lambda x: K.min(x, axis=1, keepdims=True) / \
                                                (K.max(x, axis=1, keepdims=True) + 1e-5))
            similarity_tensors.append(similarity_layer(num_concat))
    
    # create input tensors from _isna attributes
    input_isna_tensors = list()
    if make_isna:
        for side, column in it.product(sides, columns):
            input_isna_tensors.append(Input(shape=(1,)))
    
    num_dense_inputs = len(similarity_tensors) + len(input_isna_tensors)
    print('Number of inputs to dense layer: {}'.format(num_dense_inputs))
    # concatenate similarity tensors with isna_tensors.
    concatenated_tensors = Concatenate(axis=-1)(similarity_tensors + \
                                                input_isna_tensors)
    
    # create dense layers starting with concatenated tensors
    dense_tensors = [concatenated_tensors]
    for n_nodes in dense_nodes:
        dense_tensor = Dense(n_nodes, activation='relu')(dense_tensors[-1])
        if batch_norm and dropout:
            dense_tensor_bn = BatchNormalization()(dense_tensor)
            dense_tensor_dropout = Dropout(dropout)(dense_tensor_bn)
            dense_tensors.append(dense_tensor_dropout)
        else:
            dense_tensors.append(dense_tensor)
        dense_tensors.pop(0)
        
    output_tensors = Dense(2, activation='softmax')(dense_tensors[-1])
    
    product = list(it.product(sides, columns))
    model = Model([input_tensors[s][tc] for s, tc in product] + input_isna_tensors,
                  [output_tensors])
    
    return tuple([model] + list(packaged_data.values()) + [y_train, y_val, y_test])