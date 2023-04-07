import pandas as pd
import numpy as np
import os.path

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

df = pd.read_csv('object.csv')

# since attr1 and attr2 are linearly dependent, we can remove one of them
df = df.drop(columns = ['attr2'])
df.loc[df["object"] == "object1", 'object'] = 0
df.loc[df["object"] == "object2", 'object'] = 1
df["object"] = pd.to_numeric(df["object"])

attr_list = ['attr1','attr2','attr3','attr4','attr5']

# column-wise normalization
df=(df-df.min())/(df.max()-df.min())
df['object'] = df['object'].astype('int')

feature_df = df[['attr1', 'attr3', 'attr4', 'attr5']]
X = np.array(feature_df)
y = np.array(df['object'])

# train/validation/testing = 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

# lists of hyper-parameters
exps    = range(-5, 16)
C       = [2**exp for exp in exps]
exps    = range(-15, 6)
Gamma   = [2**exp for exp in exps]
Degrees = [2, 3, 4, 5]
Kernels = ['rbf', 'poly']

# convert tuple into string
def convertTuple(tup):
    to_return = ''
    for item in tup:
        to_return = to_return + str(item)
    return to_return

# extract hyper-parameters from a string
class score_and_params():
    def __init__(self, string):
        if ' C ' not in string:
            self.C = 0
        else:
            self.C = float(string.split(' C ')[1].split('=')[0].strip())
        if 'gamma' not in string:
            self.gamma = 0
        else:
            self.gamma = float(string.split('gamma')[1].split('=')[0].strip())
        if 'score' not in string:
            self.score = 0
        else:
            self.score = float(string.split('score')[1].split('=')[0].strip())
        if 'kernel' not in string:
            self.kernel = 'poly'
        else:
            self.kernel = string.split('kernel')[1].split('=')[0].strip()
        if 'degree' not in string:
            self.degree = -1
        else:
            self.degree = int(string.split('degree')[1].split('=')[0].strip())
            
# extract the hyper-parameters save in the .txt file
def get_old_scores(filename):
    accuracy  = score_and_params('')
    precision = score_and_params('')
    recall    = score_and_params('')
    f_measure = score_and_params('')
    AUC       = score_and_params('')
    
    if not os.path.exists(filename):
        return accuracy, precision, recall, f_measure, AUC
    
    with open(filename, 'r') as f:
        for row in f:
            if 'accuracy'  in row: accuracy  = score_and_params(row)
            if 'precision' in row: precision = score_and_params(row)
            if 'recall'    in row: recall    = score_and_params(row)
            if 'f_measure' in row: f_measure = score_and_params(row)
            if 'AUC'       in row: AUC       = score_and_params(row)

        return accuracy, precision, recall, f_measure, AUC
        
# if the new_score is better (higher) than the old_score, the function returns the string with the corresponding hyper-parameters
def update_scores(score_name, new_score, new_kernel, new_degree,  new_gamma, new_C, old_score, old_kernel, old_degree, old_gamma, old_C):
    if new_score > old_score:
        return convertTuple((' = ', score_name, ' = score ',new_score  ,' = kernel ',new_kernel ,' = degree ', new_degree,' = gamma ',new_gamma  ,' = C ',new_C, ' = \n'))
    return convertTuple((' = ', score_name, ' = score ',old_score ,' = kernel ', old_kernel ,' = degree ', old_degree,' = gamma ',old_gamma  ,' = C ',old_C, ' = \n'))

# location where the best score is saved during hyper-parameters fine-tuning
scores_filename = 'SVM_scores.txt'

# we loop through the hyper-parameters to train and evaluate a SVM classifier
for kernel in Kernels:
    for degree in Degrees:
        for c in C:
            for gamma in Gamma:

                scores_list=[]

                classifier = svm.SVC(kernel = kernel, gamma = gamma, C = c, degree=degree)
                classifier.fit(X_train,y_train)
                y_pred = classifier.predict(X_test)

                accuracy  = accuracy_score(y_test,y_pred)
                precision = precision_score(y_test,y_pred)
                recall    = recall_score(y_test,y_pred)
                f_measure = f1_score(y_test,y_pred)
                AUC       = roc_auc_score(y_test,y_pred)

                old_accuracy, old_precision, old_recall, old_f_measure, old_AUC = get_old_scores(scores_filename)

                # this score list contains the best scores and the related parameters after the comparison between old and new evaluation scores
                scores_list.append(update_scores('accuracy', accuracy, kernel, degree, gamma, c, old_accuracy.score, old_accuracy.kernel,old_accuracy.degree,old_accuracy.gamma, old_accuracy.C))
                scores_list.append(update_scores('precision', precision, kernel,degree, gamma, c, old_precision.score, old_precision.kernel,old_precision.degree,old_precision.gamma, old_precision.C))
                scores_list.append(update_scores('recall', recall, kernel,degree,  gamma, c, old_recall.score, old_recall.kernel,old_recall.degree,old_recall.gamma, old_recall.C))
                scores_list.append(update_scores('f_measure', f_measure, kernel,degree,  gamma, c, old_f_measure.score, old_f_measure.kernel,old_f_measure.degree,old_f_measure.gamma, old_f_measure.C))
                scores_list.append(update_scores('AUC', AUC, kernel,degree,  gamma, c, old_AUC.score, old_AUC.kernel,old_AUC.degree,old_AUC.gamma, old_AUC.C))

                with open(scores_filename, 'w') as f:
                    for row in scores_list:
                        f.write(row)

