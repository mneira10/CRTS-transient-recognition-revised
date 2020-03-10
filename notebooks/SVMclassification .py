
# coding: utf-8

# In[1]:


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
from sklearn.svm import SVC
from joblib import dump, load


# In[2]:


DATA_PATH = './FixedData/'


# In[13]:


train_data = pd.read_pickle(DATA_PATH + 'Binary/train.pkl')

train_data = train_data.drop(['Class','ObsCount'],axis=1)

test_data = pd.read_pickle(DATA_PATH + 'Binary/test.pkl')
test_data = test_data.drop(['Class','ObsCount'],axis=1)


# # Binary Classification 
# Transients and non-transients 

# In[7]:


#hypermarameters that gridsearch will optimize
def rf():
    params = {
        'kernel': ['linear','poly', 'rbf', 'sigmoid'],
        'C': [2**x for x in range(-3,6,4) ],
        'gamma': [2**x for x in range(-3,6,4) ]
    }
    return SVC(random_state=0, class_weight='balanced'), params

#metrics to be analized
def scorers():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1_score': make_scorer(f1_score, average='weighted')
               }
    return scoring


# In[ ]:


recall_scores = []


#learning 
model,params = rf()
grid_search = GridSearchCV(model, params, cv=StratifiedKFold(2), scoring=scorers(),
                           refit='f1_score', return_train_score=True,verbose=100,n_jobs=1)
grid_search.fit(train_data[train_data.columns[:-1]], train_data.target)

# Copy classifier 
clf = grid_search

#see performance on test set
scores = precision_recall_fscore_support(
        all_test.target, clf.predict(all_test[feats]), average='weighted')
recall_scores.append(scores)


# In[ ]:


recall_scoresnp = np.array(recall_scores)

print("Precision: {:.4f}".format(np.mean(recall_scoresnp[:,0])))
print("Recall:    {:.4f}".format(np.mean(recall_scoresnp[:,1])))
print("F-score:   {:.4f}".format(np.mean(recall_scoresnp[:,2])))


# In[29]:


cm = confusion_matrix(test_data.target, clf.predict(test_data[test_data.columns[:-1]])).transpose()
print(cm)


# In[30]:


def fMeasure(precision,recall):
    return 2*precision*recall/(precision+recall)


# In[31]:


precisionNon = cm[0][0]/(cm[0][0]+cm[0][1])
recallNon = cm[0][0]/(cm[0][0]+cm[1][0])
precisionT = cm[1][1]/(cm[1][1]+cm[1][0])
recallT = cm[1][1]/(cm[1][1]+cm[0][1])


# In[33]:


print("Precision of transients:     {:.4f}".format(precisionT))
print("Recall of transients:        {:.4f}".format(recallT))
print("F-measure of transients:     {:.4f}".format(fMeasure(precisionT,recallT)))
print("Precision of non-transients: {:.4f}".format(precisionNon))
print("Recall of non-transients:    {:.4f}".format(recallNon))
print("F-measure of non-transients: {:.4f}".format(fMeasure(precisionNon,recallNon)))


# In[ ]:


#https://scikit-learn.org/stable/modules/model_persistence.html
dump(clf, 'binarySVM.joblib') 


# # 8-Class clasification

# In[3]:


# main 6 transient classes
labels = ['SN', 'CV', 'AGN', 'HPM', 'Blazar', 'Flare']


# In[4]:


# func to map labels to integer values
def manualFact(lab):
    labels = ['SN', 'CV', 'AGN', 'HPM', 'Blazar', 'Flare','Other','non-transient']
    return labels.index(lab)


# In[11]:


train_data = pd.read_pickle(DATA_PATH + '8Class/train.pkl')

train_data = train_data.drop(['Class','ObsCount'],axis=1)

test_data = pd.read_pickle(DATA_PATH + '8Class/test.pkl')
test_data = test_data.drop(['Class','ObsCount'],axis=1)


# In[13]:


recall_scores = []

model,params = rf()
grid_search = GridSearchCV(model, params, cv=StratifiedKFold(2), scoring=scorers(),
                           refit='f1_score', return_train_score=True,verbose=100,n_jobs=-1)
grid_search.fit(train_data[train_data.columns[:-1]], train_data.target)
# Copy classifier
clf = grid_search

scores = precision_recall_fscore_support(
        all_test.target, clf.predict(all_test[feats]), average='weighted')


recall_scores.append(scores)


# In[39]:


recall_scoresnp = np.array(recall_scores)

print("Precision: {:.4f}".format(np.mean(recall_scoresnp[:,0])))
print("Recall:    {:.4f}".format(np.mean(recall_scoresnp[:,1])))
print("F-score:   {:.4f}".format(np.mean(recall_scoresnp[:,2])))


# In[14]:


confMatr = confusion_matrix(test_data.target, clf.predict(test_data[test_data.columns[:-1]])).transpose()
print(confMatr)
# 'SN', 'CV', 'AGN', 'HPM', 'Blazar', 'Flare','Other','non-transient'


# In[42]:


normedMatrix = confMatr.copy()
# print(normedMatrix)
for i in range(len(normedMatrix)):
#     print(normedMatrix[:,i]/normedMatrix[:,i].sum())
    normedMatrix[:,i] = normedMatrix[:,i]/normedMatrix[:,i].sum()
#     print(normedMatrix[:,i])
# np.set_printoptions(suppress=True)
print(normedMatrix)
# np.set_printoptions(suppress=False)


# In[43]:


newMatr = []
for i in range(len(confMatr)):
    prec = confMatr[i][i]/(sum(confMatr[i,:]))
    rec = confMatr[i][i]/(sum(confMatr[:,i]))
    newMatr.append([prec,rec,fMeasure(prec,rec),sum(confMatr[:,i])])


# In[44]:


np.set_printoptions(suppress=True)
print('    Precision        Recall         F-score      Cover')
print(np.array(newMatr))
# np.set_printoptions(suppress=False)


# In[ ]:


dump(clf, '8ClassSVM.joblib') 

