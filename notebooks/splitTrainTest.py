import pandas as pd
import numpy as np


def loadData():
    DATA_PATH = '../data/features/newNegsTrueSet/'
    negsFilePath = 'new_NT.csv'
    transientsFilePath = 'T.csv'

    origTransients = pd.read_csv(DATA_PATH+transientsFilePath)
    origNonTransients = pd.read_csv(DATA_PATH+negsFilePath)

    origTransients = origTransients.set_index(['ID', 'copy_num'])
    origNonTransients = origNonTransients.set_index(['ID', 'copy_num'])

    return origTransients, origNonTransients


def manualFact(lab):
    labels = ['SN', 'CV', 'AGN', 'HPM', 'Blazar',
              'Flare', 'Other', 'non-transient']
    return labels.index(lab)


def splitTrainTest(classificationProblem):
    '''
        Load transient and non transient features

        Create target classes and split into train and test sets for
        binary and 8 class classification problems

        Note that non transients have not been oversampled

        Also note that all classes in the training set MUST BE
        EQUALLY REPRESENTED to avoid any bias at training.

        IMPORTANT: set seed to replicate splits
    '''
    # input verification
    assert classificationProblem in ['binary', '8class'], \
        '''classificationProblem not in ['binary','8class']'''

    # set seed
    np.random.seed(0)

    # load data
    T, NT = loadData()

    if classificationProblem == 'binary':
        # binary classification problem

        # set classes
        T['Class'] = 1
        NT['Class'] = 0

        print('binary case, NT: {} T {}'.format(NT.shape[0],T[T.index.get_level_values('copy_num')==0].shape[0]))

        # see which df is the smallest

        small = T if T[T.index.get_level_values('copy_num')==0].shape[0] < NT.shape[0] else NT
        big = NT if T[T.index.get_level_values('copy_num') == 0].shape[0]< NT.shape[0] else T
        
        assert T[T.index.get_level_values('copy_num')==0].shape[0] == len(T.index.get_level_values('ID').unique())
        
        # split into train and test
        uniqueSmallIds = small.index.get_level_values('ID').unique()

        testSmallIds = np.random.choice(
            uniqueSmallIds, int(len(uniqueSmallIds)*0.25), replace=False)

        # remove oversamples from test set
        smallTest = small[(small.index.get_level_values('ID').isin(testSmallIds))
                          & (small.index.get_level_values('copy_num') == 0)]

        smallTrain = small[~small.index.get_level_values(
            'ID').isin(testSmallIds)]

        smallTest['set'] = 'test'
        smallTrain['set'] = 'train'

        # randomly get the same number of samples in train set
        # no copies here, no need to get uniques
        bigTest = big.sample(n=int(len(big)*0.25), replace = False)
        
        bigIds = big.index.get_level_values('ID')

        assert len(bigIds)==len(big)


        bigTrain = big[~big.index.get_level_values('ID').isin(bigTest.index.get_level_values('ID'))] #np.random.choice(
            #bigIds, min(len(smallTrain),len(bigIds)), replace=False)

        print('the minimum: {} lensmallTrain {} lenbigids {}'.format(min(len(smallTrain),len(bigTrain)),len(smallTrain),len(bigTrain)))
        
        smallTrain = smallTrain.sample(n=min(len(smallTrain),len(bigTrain)), replace=False)
        bigTrain = bigTrain.sample(n=min(len(smallTrain),len(bigTrain)),replace=False)

        print('the minimum: {} lensmallTrain {} lenbigids {}'.format(min(len(smallTrain),len(bigTrain)),len(smallTrain),len(bigTrain)))


        #bigTrain = big[big.index.get_level_values('ID').isin(bigTrainIDs)].sample(n=len(smallTrain), replace=False) # sampling here is to mix up samples
        #bigTest = big[(~big.index.get_level_values('ID').isin(bigTrainIDs)) & (
         #   big.index.get_level_values('copy_num') == 0)]

        bigTrain['set'] = 'train'
        bigTest['set'] = 'test'

        train = pd.concat([smallTrain, bigTrain]).sample(frac=1)
        test = pd.concat([smallTest, bigTest]).sample(frac=1)

        return test, train

    elif classificationProblem == '8class':
        # 8 class classification problem
        transientLabels = ['SN', 'CV', 'AGN', 'HPM', 'Blazar', 'Flare']

        # set classes
        T['Class'] = T['Class'].apply(
            lambda x: x if x in transientLabels else 'Other')

        T['Class'] = T['Class'].apply(lambda x: manualFact(x))
        NT['Class'] = NT['Class'].apply(lambda x: manualFact(x))

        TTest = pd.DataFrame()

        # # --------------------------------------
        # # get least represented class

        # minNum = 80000000

        # for l in transientLabels:
        #     numL = manualFact(l)

        #     classDf = T[T['Class'] == numL]
        #     assert len(classDf.Class.unique()) == 1
        #     uniqueIDs = classDf.index.get_level_values('ID').unique()

        #     numUniqueIds = len(uniqueIDs)

        #     if numUniqueIds < minNum:
        #         print(l, 'has fewer unique Ids with',
        #               numUniqueIds, numUniqueIds*0.25)
        #         minNum = numUniqueIds

        # print('minimum number of class ids: {}'.format(minNum))
        # # --------------------------------------

        for label in ['SN', 'CV', 'AGN', 'HPM', 'Blazar',
                      'Flare', 'Other']:
            print('Processing label', label)

            numericLabel = manualFact(label)

            # gert class data
            TclassDf = T[T['Class'] == numericLabel]

            # get class ids
            TclassUniqueIds = TclassDf.index.get_level_values('ID').unique()
            print(label, 'has', len(TclassUniqueIds), 'unique ids')

            # get test ids
            TTestClassIds = np.random.choice(
                TclassUniqueIds, int(len(TclassUniqueIds)*0.25), replace=False)

            # get test df for class
            TTestClassDf = \
                TclassDf[
                    (TclassDf.index.get_level_values('ID').isin(TTestClassIds))
                    &
                    (TclassDf.index.get_level_values('copy_num') == 0)
                ]

            assert len(TTestClassDf) == int(len(TclassUniqueIds)*0.25)
            # append to test df
            TTest = pd.concat([TTest, TTestClassDf])

        tempTTrain = T[~T.index.get_level_values(
            'ID').isin(TTest.index.get_level_values('ID'))]

        assert tempTTrain.index.get_level_values('ID').isin(
            TTest.index.get_level_values('ID')).sum() == 0

        minLcCount = tempTTrain.groupby('Class').count()['amplitude'].min()

        print('getting', minLcCount, 'light curves')
        TTrain = pd.DataFrame()

        # equal amount of lcs for each class in training set
        for label in ['SN', 'CV', 'AGN', 'HPM', 'Blazar',
                      'Flare', 'Other']:
            temp = tempTTrain[tempTTrain.Class == manualFact(label)]
            temp = temp.sample(n=minLcCount, random_state=1)
            TTrain = pd.concat([TTrain, temp])

        TTest['set'] = 'test'
        TTrain['set'] = 'train'

        # randomly get the same number of samples in train set
        # no copies here, no need to get uniques
        NTIds = NT.index.get_level_values('ID').unique()
        assert len(NTIds) == len(NT)

        NTTrainIDs = np.random.choice(
            NTIds, minLcCount, replace=False)

        NTTrain = NT[NT.index.get_level_values('ID').isin(NTTrainIDs)]
        NTTest = NT[~NT.index.get_level_values('ID').isin(NTTrainIDs)]

        NTTrain['set'] = 'train'
        NTTest['set'] = 'test'

        train = pd.concat([TTrain, NTTrain]).sample(frac=1)
        test = pd.concat([TTest, NTTest]).sample(frac=1)

        return test, train
