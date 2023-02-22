import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, MinMaxScaler
import re
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.metrics import mean_squared_error
from pandas import Series
from math import sqrt, pow
from numpy import concatenate, subtract
from pandas import DataFrame
from pandas import concat
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn import model_selection
import os
import joblib
from tensorflow.keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest
import scipy.stats as st
from pandas import Series
from tensorflow.keras.preprocessing.text import Tokenizer
from pandas import Series
import joblib 
import pickle as pkl
from datetime import datetime, timedelta
from sklearn.preprocessing import normalize, MinMaxScaler
from math import sqrt, pow
from tensorflow.keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest

# ================= get the vocabulary set ==================
def vocabulary_generate(fd, key_para_dict_filename, key_para_dict_index_filename):
    '''
    :param fd:  pandas dataframe with the log key column in it is hashed values
    :return: fd_id: copied fd dataframe, in order to protect the original data
             key_para_dict: the format is {Exx:[textual parameter 1],[textual parameter 2],...}
    '''

    key_para_dict, key_para_dict_index = {}, {}
    log_key_sequence, key_name_dict, K = key_to_EventId(fd)
    fd_id = fd.copy()
    # switch the key and value in a dict
    key_name_dict_rev = dict((value,key) for key,value in key_name_dict.items())
    # mapping the value to keyID
    fd_id['log key'] = fd_id['log key'].map(key_name_dict_rev)

    uni_log_key_id = list(set(fd_id['log key']))

    parameters, parameters_index = [], []

    for i in range(len(uni_log_key_id)):
        # get all the parameters with the same eventID
        parameters = fd_id[fd_id['log key'] == uni_log_key_id[i]]['parameter value vector']
        print(parameters)
        para_index = parameters.index.values
        # filter special characters
        #parameters = template_filter(parameters)
        # transform the series object to list type
        parameters_index = list(parameters)
        parameters_index.insert(0,str(para_index))
        key_para_dict[uni_log_key_id[i]] = parameters.values[:]
        key_para_dict_index[uni_log_key_id[i]] = parameters_index

    # padding nan to object without enough length
    df_dict_para = pd.DataFrame(dict([(k,Series(v)) for k,v in key_para_dict.items()]))
    df_dict_para_index = pd.DataFrame(dict([(k,Series(v)) for k,v in key_para_dict_index.items()]))

    df_dict_para.to_csv(key_para_dict_filename,index = False, header=key_para_dict.keys())
    # add the index of original index to the dict
    df_dict_para_index.to_csv(key_para_dict_index_filename, index = False, header=key_para_dict_index.keys())
    return key_para_dict, fd_id


def key_to_EventId(df):

    '''
    :param df: normaly, the log key column in df is hashed values
    :return: log_key_sequence: the column of log key
             key_name_dict: format is {Exx: SRWEDFFW(hashed value),...}
             K: the number of unique log key events
    '''

    df = df.copy()
    log_key_sequence = df['log key']
    log_key_sequence = list(log_key_sequence)
    # get the unique list
    items = set(log_key_sequence)
    # define the total number of log keys
    K = None
    K = len(items)
    print("the length of log_key_sequence is:", len(items))
    key_name_dict = {}

    for i, item in enumerate(items):
        # items is a set
        # columns are the lines of log key sequence
        for j in range(len(log_key_sequence)):
            if log_key_sequence[j] == item:
                name = 'E' + str(i)
                # log_key_sequence[j]='k'+str(i)
                key_name_dict[name] = log_key_sequence[j].strip('\n')

    return log_key_sequence, key_name_dict, K


def tokens_generate(key_para_dict):
    '''
    :param key_para_dict: the format is {Exx:[textual parameter 1],[texual parameter 2],...}
    :return: tokens: all the word tokens in the parameter value vector column
    '''
    text = []
    for key, value in key_para_dict.items():
        # extract the time part from values
        for i in range(len(value[:])):
            if value[i].split(',')[1:] == []:
                break
            else:
                value[i] = re.sub('[\[|\]|\'|\|\s+|\.|\-]', '', str(value[i])).split(',')
                if value[i] == ['']:
                    break
                else:
                    text.append([var for var in value[i][1:]])

    # get the text for token_nize
    tokens = []
    for i in range(len(text)):
        for j in range(len(text[i])):
            tokens.append(text[i][j])
    # delete the blank value
    tokens = [var for var in tokens if var]
    tokens = set(tokens)

    return tokens


def token_dict(tokens, tokens_dict_filename):
    '''
    :param tokens: all the word tokens in the parameter value vector column
    :return: token_encode_dict: the format is ['fawjeiajet';[32,45,65,..],...]
    '''

    # build the dict about different value
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)
    encoded_texts = tokenizer.texts_to_sequences(tokens)
    # build the dict with tokens --> encoded_texts
    token_encode_dict = {}
    #print(tokens)
    #print(encoded_texts)
    for token, encoded_text in zip(tokens, encoded_texts):
        token_encode_dict[token] = encoded_text
    joblib.dump(token_encode_dict, tokens_dict_filename)

    return token_encode_dict

def split_vectors(fd_id, filename):
    '''
    :param fd_id: copied fd dataframe, in order to protect the original data
    :param filename: the position to store csv
    :return: fd_id: csv with parameter value vector splitted into various columns according to the max length of vector
            list_name: the format is: value0, value1, value2, ....
    '''
    list_length = []
    for var in fd_id['parameter value vector']:
        list_length.append(len(var.split(',')))
    # max(list_length) ---- 6
    # list_length
    list_name = []
    for i in range(max(list_length)):
        list_name.append('value' + str(i))
    fd_id[list_name] = fd_id['parameter value vector'].str.split(",", expand=True)
    # [var for var in fd_id['value15'] if var]
    # fd_id
    for name in list_name:
        for var in range(len(fd_id[name])):
            # we should use fd_id[x] to rewrite value in
            if fd_id[name][var] != None:
                fd_id[name][var] = re.sub("[\[|\]|']|\s+|\.|\-", '', fd_id[name][var])
    fd_id.to_csv(filename, index=False)

    return fd_id, list_name

def map_vectors(fd_id, list_name, filename, token_encode_dict):
    '''
    :param fd_id: csv with parameter value vector splitted into various columns according to the max length of vector
    :param list_name: the format is: value0, value1, value2, ....
    :return: fd_value: csv with textual values in parameter value vector replaced by numerical values
    '''
    fd_value = fd_id
    for var in range(1,len(list_name)):
        fd_value[list_name[var]] = fd_value[list_name[var]].map(token_encode_dict)
    fd_value.to_csv(filename, index = False)

    return fd_value

def integrate_lines(fd_value, list_name):
    fd_value['ColumnX'] = fd_value[fd_value.columns[3:len(list_name)]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
    fd_value = fd_value.drop(['parameter value vector'], axis=1)
    fd_value = fd_value.drop(list_name, axis=1)

    return fd_value

def delete_repeated_line(fd_value,filename):
    fd_value['parameter value vector'] = fd_value['ColumnX']
    fd_value = fd_value.drop(['ColumnX'], axis=1)
    fd_value.to_csv(filename, index=False)
    return fd_value
def log_vectors(fd_values, filename):
    '''
    :param fd_values: csv with numerical values in a single parameter value vector column
    :return: key_num_para_dict: the format is {Exx:[numerical parameter 1],[numerical parameter 2],...}
    '''
    uni_log_key_id = list(set(fd_values['log key']))

    parameters = []

    key_num_para_dict = {}
    for i in range(len(uni_log_key_id)):
        # get all the parameters with the same eventID
        parameters = fd_values[fd_values['log key'] == uni_log_key_id[i]]['parameter value vector']
        # the format will be E88: [xx,xx,xx],[xf,er,et],[re,tet,tet],...
        key_num_para_dict[uni_log_key_id[i]] = parameters.values[:]
    # plt.hist(key_para_dict['E88'], bins=15)
    # plt.ylable('Counter')
    # plt.xlabel('Variables')
    # plt.title('Single Log Key Vectors')
    print("the num para dict is:",key_num_para_dict)

    df_dict_num_para = pd.DataFrame(dict([(k, Series(v)) for k, v in key_num_para_dict.items()]))
    # df_dict_num_para.to_csv('../Dataset/Linux/Malicious_Separate_Structured_Logs/key_num_para_dict.csv', index= False, header=key_num_para_dict.keys())
    df_dict_num_para.to_csv(filename, index= False, header=key_num_para_dict.keys())
    return key_num_para_dict

# define the module to transform str into matrix
# the string is like: '10635,[21, 85, 16, 18],[21, 85, 16, 18, 307, 308, 1],[356],[424],[207]'
def str_array(dict, eventID, filename, i):
    '''
    :param dict: the format is key:[numerical parameter 1],[numerical parameter 2],..
    :param eventID: Exx
    :return: saved matrix for unique event
    '''
    lists = dict[eventID]
    #print(f'\n ------------------>  Current Key Param dict list is : >>>>>>  {lists}')
    list_string = []
    pattern = '\d+'
    numx = len(lists)
    #print(f'Length of lists is: {numx}')
    #numy = len(re.findall(pattern, lists[0]))
    numy = max([len(re.findall('\d+',i)) for i in lists])
    #print(f'Length of numerical pattern is:  {numy} where lists has shape: {lists.shape}')
    list_array = np.empty(shape=[0, numy])
    #print(f'Shape of list array: {list_array.shape}')
    for string in lists:
        #print('----------Printing List----------')
        # matching all digits
        list_string = re.findall(pattern, string)
        #print(f'Numerical Patter out of list: {list_string}')
        
        # transform the str into int
        list_string = [int(var) for var in list_string]
        #print(f'mapped to int: {list_string}')
        
        list_string = np.array(list_string)
        #print(f'mapped into numpy array: {list_string}')
        
        if len(list_string) < numy:
            list_string = np.pad(list_string, (0,numy-len(list_string)), 'mean')
        
        #print(f'Shape of list_string :{list_string.shape} == shape of empty array(list_array) {list_array.shape}')
        # concatenate multi lines
        try:
            list_array = np.append(list_array, [list_string], axis = 0)
            #print(f'\n List appended {list_array} \n')
        except Exception as e:
            print("there is an error like:", e)
            pass
    print(f'\n',list_array)
    #np.save('npy/'+ eventID +'.npy', list_array)
    if os.path.exists(filename):
        np.save(filename + eventID+'.npy', list_array)
    else:
        os.mkdir(filename)
        np.save(filename + eventID+'.npy', list_array)

def training_data_generate(matrix, n_steps):
    '''
    :param matrix: the paramter value vectors for a single log key
    :param n_steps_in: the length of sequence, which depends on how long the matrix is
    :param n_steps_out: always one, we just need one really parameter vector
    :return:
    '''
    X, Y = list(), list()
    for i in range(matrix.shape[0]):
        # find the end of this pattern
        end_ix = i+n_steps
        # check whether beyond the dataset
        if end_ix > matrix.shape[0]-1:
            break
        seq_x, seq_y = matrix[i:end_ix,:], matrix[end_ix,:]
        X.append(seq_x)
        Y.append(seq_y)
    X, Y = np.array(X), np.array(Y)
    print("the shape of X is:",X.shape)
    return X, Y

def LSTM_model(trainx, trainy):
    # use the train
    model = Sequential()
    model.add(LSTM(120, activation = 'relu', return_sequences = True, input_shape=(trainx.shape[1], trainx.shape[2])))
    print("the train x is {} and its shape is {}".format(trainx, trainx.shape))
    model.add(LSTM(120, activation = 'relu'))
    model.add(Dense(trainx.shape[2]))
    model.compile(loss='mse', optimizer='adam')
    # model.fit(trainx, trainy, epochs = 50, verbose=2, callbacks=[callbacks])
    model.fit(trainx, trainy, epochs=60, verbose=2)
    model.summary()
    #joblib.dump(model,'model.pkl')
    model.save('parameter_model')
    return model
def mean_squared_error_modified(y_true, y_pred):
    d_matrix = subtract(y_true, y_pred)
    print("the d_matrix is:", d_matrix)
    means = []
    for i in range(d_matrix.shape[1]):
        means.append(np.mean(d_matrix[:, i] * d_matrix[:, i], axis=-1))
    print("the means are:", means)
    return np.mean(means), means

def confidence_intervial(confidence, mses_list):
    # define the intervial tuple
    intervial = None
    intervial = st.t.interval(confidence, len(mses_list)-1, loc=np.mean(mses_list), scale=st.sem(mses_list))

    return intervial

def anomaly_report(mses_list,file_number):
    # here we use the max value as the threshold
    confidence_intervial_fp1 = confidence_intervial(0.98, mses_list)
    # it is for the false positive detection
    threshold1 = confidence_intervial_fp1[1]
    confidence_intervial_fp2 = confidence_intervial(0.99, mses_list)
    # it is for the false positive detection
    threshold2 = confidence_intervial_fp2[1]
    confidence_intervial_an = confidence_intervial(0.999, mses_list)
    # it is for the anomaly detection
    threshold3 = confidence_intervial_an[1]
    # record the potential anomaly logs
    suspicious_logs = []
    # record the false positive logs
    fp_logs = []
    for i in range(len(mses_list)):
        if mses_list[i] > threshold3:
            print('The {}th log in matrix {} is suspiciously anomaly'.format(i, file_number[0]))
            suspicious_logs.append(i)
        elif mses_list[i] > threshold1:
            print('The {}th log in matrix {} is false positive'.format(i, file_number[0]))
            fp_logs.append(i)
        else:
            continue
    return threshold1, threshold2, threshold3, suspicious_logs, fp_logs
