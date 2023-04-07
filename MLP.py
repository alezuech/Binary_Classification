import pandas as pd
import numpy as np
import os.path
import keras.backend as K
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import SGD

df = pd.read_csv('object.csv')

# since attr1 and attr2 are linearly dependent, we can remove one of them
df = df.drop(columns = ['attr2'])
df.loc[df["object"] == "object1", 'object'] = 0
df.loc[df["object"] == "object2", 'object'] = 1
df["object"] = pd.to_numeric(df["object"])

df_a = np.array(df)

# m=number of samples, n=class + number of features
m, n = df_a.shape

np.random.seed(0)
np.random.shuffle(df_a)

# number of training samples
train_samples=int(m*0.8)

# Creation of training and testing datasets
data_train = df_a[0:train_samples].T
X_train=data_train[1:]
Y_train=data_train[0].astype(int)

data_test = df_a[train_samples:].T
X_test=data_test[1:]
Y_test=data_test[0].astype(int)

# Column-wise normalization of the features
for i in range(n):
    df_a[:,i] = (df_a[:,i]-np.min(df_a[:,i]))/(np.max(df_a[:,i])-np.min(df_a[:,i]))

# convert tuple of strings into single string
def convertTuple(tup):
    to_return = ''
    for item in tup:
        to_return = to_return + str(item)
    return to_return

# get float variable from a string
def get_value(string, value_name):
    return float(string.split(value_name)[1].split('--')[0].strip())

# get hyper-parameters from a string
def extract_params(string):
    h_u = get_value(string, 'h_u:')
    h_l = get_value(string, 'h_l:')
    dr  = get_value(string, 'dr:' )
    b_s = get_value(string, 'b_s:')
    l_r = get_value(string, 'l_r:')
    mo  = get_value(string, 'mo:' )
    return h_u, h_l, dr, b_s, l_r, mo

# get hyper-parameters from a file
def params_from_file(file_name, score_name):
    if not os.path.exists(file_name): return 0, 0, 0, 0, 0, 0

    with open(file_name, 'r') as f:
        for row in f:
            if score_name in row: return extract_params(row)        

# get float scores from a .txt file
def score_from_file(file_name):
    accuracy, precision, recall, f1_metric, auc = 0,0,0,0,0
    
    if not os.path.exists(file_name):
        return accuracy, precision, recall, f1_metric, auc
    
    def get_score(string):
        return float(string.split('\',')[1].replace(']','').strip())
    with open(file_name, 'r') as f:
        for row in f:
            if 'Accuracy' in row:
                accuracy  = get_score(row)
            if 'Precision' in row:
                precision = get_score(row)
            if 'Recall' in row:
                recall    = get_score(row)
            if 'f1_metric' in row:
                f1_metric = get_score(row)
            if 'AUC' in row:
                auc       = get_score(row)
    
        return accuracy, precision, recall, f1_metric, auc


def f1_metric_function(y_true, y_pred):
    true_positives      = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives  = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall =    true_positives / (possible_positives + K.epsilon())
    f1_val =    2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# given the hyper-parameters, an MLP is built and trained, and the evaluation metrics are returned
def train_and_evaluate(hidden_units, hidden_layers, dropout_rate, batch_size, epochs, learning_rate, momentum):
    
    # decaying learning rate
    decay_rate = learning_rate / epochs

    # stochastic gradient descent
    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    
    model = Sequential()
    model.add(Dense(units=hidden_units, activation='relu', kernel_initializer='normal', input_dim=X_train.shape[0]))
    if hidden_layers!=0:
        for i in range(hidden_layers):
            model.add(Dense(units=hidden_units, activation='relu'))
            model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', 
                  optimizer=sgd, 
                  metrics=['accuracy',
                  tf.keras.metrics.Precision(),                      
                  tf.keras.metrics.Recall(),
                  f1_metric_function,                                  
                  tf.keras.metrics.AUC()])

    model.fit(X_train.T, 
              Y_train.T, 
              epochs=epochs, 
              batch_size=batch_size, 
              verbose=0,
              validation_data=(X_test.T, Y_test.T))
    
    to_return = model.evaluate(X_test.T, Y_test.T, verbose=0)
    del model
    return to_return

# lists of hyper-parameters to tune
hidden_units   = [2, 4, 6, 8]
hidden_layers  = [0, 1, 2, 3, 4]
dropout_rates  = [0.1, 0.2, 0.3]
batch_sizes    = [32, 64, 128]
epochs         = 200
learning_rates = [0.1, 0.01]
momentums      = [0.1, 0.2, 0.3]

# get string containing parameters
def params_tuple(params_tuple):
    p_list=list(params_tuple)
    h_u,h_l,dr,b_s,l_r,mo = p_list[0],p_list[1],p_list[2],p_list[3],p_list[4],p_list[5],
    params = ('h_u: ',h_u,' ---  h_l: ',h_l,' ---  dr: ',dr,' ---  b_s: ',b_s,
                              ' ---  l_r: ',l_r ,' ---  mo: ',mo,' ---  ')
    return convertTuple(params)

# location where the best scores and the relative hyper-parameters are saved during their fine-tuning
scores_filename = 'MLP_scores.txt'

# location where the last used hyper-parameters used during fine-tuning are saved. Can be used as a checkpoint
params_filename = 'MLP_last_model_params.txt'

# fine-tuning by looping through the hyper-parameters
for h_u in hidden_units:
    for h_l in hidden_layers:
        for dr in dropout_rates:
            for b_s in batch_sizes:
                for l_r in learning_rates:
                    for mo in momentums:
                        
                        params=params_tuple((h_u, h_l, dr, b_s, l_r, mo))
                        
                        score = train_and_evaluate(hidden_units = h_u, 
                                                   hidden_layers = h_l, 
                                                   dropout_rate = dr, 
                                                   batch_size = b_s, 
                                                   epochs = epochs, 
                                                   learning_rate = l_r, 
                                                   momentum = mo)
                        
                        acc       = score[1]
                        prec      = score[2]
                        rec       = score[3]
                        f1_metric = score[4]
                        auc       = score[5]
                        
                        # the best SCORES are extracted from the .txt file.
                        best_acc, best_prec, best_rec, best_f1_metric, best_auc=score_from_file(scores_filename)

                        # the hyper-parameters of each best score are extracted from the .txt file
                        params1 = params_tuple(params_from_file(scores_filename, 'Accuracy'))
                        params2 = params_tuple(params_from_file(scores_filename, 'Precision'))
                        params3 = params_tuple(params_from_file(scores_filename, 'Recall'))
                        params4 = params_tuple(params_from_file(scores_filename, 'f1_metric'))
                        params5 = params_tuple(params_from_file(scores_filename, 'AUC'))
                        
                        best_score = {'Accuracy': [params1,best_acc], 
                                     'Precision': [params2,best_prec], 
                                     'Recall':    [params3,best_rec], 
                                     'f1_metric': [params4,best_f1_metric], 
                                     'AUC':       [params5,best_auc]}
                        new_score = {'Accuracy':  [params,acc], 
                                     'Precision': [params,prec], 
                                     'Recall':    [params,rec], 
                                     'f1_metric': [params,f1_metric], 
                                     'AUC':       [params,auc]}

                        # the best scores and the current scores are compared
                        # if the new score is better, it is save (with its hyper-parameters) in the .txt file as new best score
                        score_name = 'Accuracy'
                        if best_acc < acc: 
                            best_score[score_name][1] = acc
                            best_score[score_name][0] = params

                        score_name = 'Precision'
                        if best_prec < prec: 
                            best_score[score_name][1] = prec
                            best_score[score_name][0] = params

                        score_name = 'Recall'
                        if best_rec < rec: 
                            best_score[score_name][1] = rec
                            best_score[score_name][0] = params
                            
                        score_name = 'f1_metric'
                        if best_f1_metric < f1_metric: 
                            best_score[score_name][1] = f1_metric
                            best_score[score_name][0] = params

                        score_name = 'AUC'
                        if best_auc < auc: 
                            best_score[score_name][1] = auc
                            best_score[score_name][0] = params

                        with open(scores_filename, 'w') as f:
                            for bs in best_score:
                                to_write = convertTuple((bs, ' ', best_score[bs],'\n'))
                                f.write(to_write)
                                
                        with open(params_filename, 'w') as f:
                            f.write(params)


