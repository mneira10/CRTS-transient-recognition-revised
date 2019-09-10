# coding: utf-8

from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # , cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix  # , classification_report
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy import integrate
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support,make_scorer
from sklearn.metrics import confusion_matrix

import pickle

DATA_PATH = '../data/features/'

tran = pd.read_csv(DATA_PATH + "T.csv")
tran = tran.set_index(['ID', 'copy_num'])

ntran = pd.read_csv(DATA_PATH + "NT.csv")
ntran = ntran.set_index(['ID', 'copy_num'])

SALT2_PATH = './chi2Salt2/'

tsalt  = pd.read_csv(SALT2_PATH + 'transient/allTransient.dat'      , names = ['ID', 'copy_num','chi2Salt2'],delimiter=' ')
ntsalt = pd.read_csv(SALT2_PATH + 'nonTransient/allNonTransient.dat', names = ['ID', 'copy_num','chi2Salt2'],delimiter=' ')

tsalt = tsalt.set_index(['ID', 'copy_num'])
ntsalt = ntsalt.set_index(['ID', 'copy_num'])


print('Original')
print('transients',tran.shape)
print('non transients',ntran.shape)

tran = tran.join(tsalt, how='inner')
ntran = ntran.join(ntsalt, how='inner')

print('SALT2')
print('transients',tran.shape)
print('non transients',ntran.shape)
print("Total amout of obects (oversampled):     {}".format(tran.shape[0]+ntran.shape[0]))
print("Total amout of obects (not oversampled): {}".format(tran[tran.index.get_level_values("copy_num")==0].shape[0]+ntran.shape[0]))

#remove unwanted columns
unwanted = [
    #'poly3_t3',
    #'poly3_t2',
    #'poly3_t1',
    #'poly4_t1',
    #'poly4_t2',
    #'poly4_t3',
    #'poly4_t4',
    #'chi2SALT2',
    ]    

tran  =  tran.drop(unwanted,axis=1)
ntran = ntran.drop(unwanted,axis=1)

#reorder for easy feature selection
# tran = tran[:-3]+tran[-1:]+tran[:-3]+tran[-3:-1]
# ntran = ntran[:-3]+ntran[-1:]+ntran[:-3]+ntran[-3:-1]
def totFeats():
    feats = np.array(list(tran.columns[:-3])+[tran.columns[-1]])
    return feats

feats = totFeats()
print("Total number of features: {}".format(len(feats)))
print()
print("The features are:")
for i,f in enumerate(feats):
    print("    "+str(i+1)+". " + f)


# # Binary Classification 
# Transients and non-transients 

def splitTrainTest(dataframe):
    #create output dataframes
    test = pd.DataFrame(columns = ["ID","copy_num"]+list(dataframe.columns))
    test = test.set_index(["ID","copy_num"])

    train = pd.DataFrame(columns = ["ID","copy_num"]+list(dataframe.columns))
    train = train.set_index(["ID","copy_num"])
    
    for uClass in dataframe.Class.unique():
        #get each class 
        classDf = dataframe[dataframe.Class == uClass]
        
        #unique ids
        ids = classDf.index.get_level_values('ID').unique()

        # randomly choose 25% of indices 

        testInd = np.random.choice(ids, int(0.25*len(ids)),replace=False)

        #get dataframes
        test = pd.concat([test,classDf[classDf.index.get_level_values('ID').isin(testInd)]])
        
        train = pd.concat([train,classDf[~classDf.index.get_level_values('ID').isin(testInd)]])

    return train,test

def balance(df):
    #start min at infinity 
    minNum = np.inf
    
    #find the class with the minimum amount of candidates
    for classElem in df.Class.unique():
        numElems = len(df[df.Class==classElem])
        if(numElems<minNum):
            minNum=numElems
    
    #create output dataframe
    ans = pd.DataFrame(columns = ["ID","copy_num"]+list(df.columns))
    ans = ans.set_index(["ID","copy_num"])
    
    #get a sample from all the classes 
    for classElem in df.Class.unique():
        ans = pd.concat([ans,df[df.Class==classElem].sample(n=minNum)])
        
    return ans

#hypermarameters that gridsearch will optimize
def rf():
    params = {
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    return RandomForestClassifier(random_state=0, class_weight='balanced'), params

#metrics to be analized
def scorers():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1_score': make_scorer(f1_score, average='weighted')
               }
    return scoring

#copy and shuffle the data
alldf = pd.concat([tran,ntran])
oversampling = alldf.copy()
oversampling = oversampling.sample(frac=1)

#map all objects that are not non-transient to transient
oversampling.Class = list(map(lambda x: 'SN' if x=='SN' else 'NSN', oversampling.Class))
#map transient and non-transient to binary values
oversampling['target'] = list(map(lambda x: 1 if x=='SN' else 0, oversampling.Class))






def getData():
    #split train test class by class
    all_train,all_test = splitTrainTest(oversampling)

    #balance the train set
    all_train= balance(all_train)

    #train indices
    trainIdx = all_train.index.get_level_values("ID").unique()

    #remove originals that have oversampled copies in train
    all_test = all_test[~all_test.index.get_level_values("ID").isin(trainIdx)]

    #remove oversampled data from test set
    all_test = all_test[all_test.index.get_level_values('copy_num') ==0 ]


    #format target variable to appropriate data type
    all_train.target= all_train.target.astype('int')
    all_test.target= all_test.target.astype('int')

    print("SN in test set:")
    print(len(all_test[all_test.Class=='SN']))
    print("NSN in test set:")
    print(len(all_test[all_test.Class!='SN']))
    return all_train, all_test

all_train, all_test = getData()


pickle.dump( all_train, open( "./cleanData/stableData/train.pkl", "wb" ) )
pickle.dump( all_test, open(  "./cleanData/stableData/test.pkl", "wb" ) )
