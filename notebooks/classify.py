from rfClassifier import RFClassifier
from svmClassifier import SVMClassifier

from os import path
import pickle
import os
from manageResults import plotFeatImportances, genMetricsAndCM
from splitTrainTest import splitTrainTest

import numpy as np
np.random.seed(0)


allFeatures = ['skew',
               'std',
               'kurtosis',
               'beyond1st',
               'stetson_j',
               'stetson_k',
               'max_slope',
               'amplitude',
               'median_absolute_deviation',
               'median_buffer_range_percentage',
               'pair_slope_trend',
               'percent_amplitude',
               'percent_difference_flux_percentile',
               'flux_percentile_ratio_mid20',
               'flux_percentile_ratio_mid35',
               'flux_percentile_ratio_mid50',
               'flux_percentile_ratio_mid65',
               'flux_percentile_ratio_mid80',
               'small_kurtosis',
               'pair_slope_trend_last_30',
               'poly1_t1',
               'poly2_t2',
               'poly2_t1',
               'poly3_t3',
               'poly3_t2',
               'poly3_t1',
               'poly4_t4',
               'poly4_t3',
               'poly4_t2',
               'poly4_t1',
               'magnitudeRatio',
               'lombScargle',
               'rcb']

disantoFeatures = ['amplitude', 'beyond1st', 'flux_percentile_ratio_mid20', 'flux_percentile_ratio_mid35', 'flux_percentile_ratio_mid50',
                   'flux_percentile_ratio_mid65', 'flux_percentile_ratio_mid80', 'poly1_t1', 'median_absolute_deviation',
                   'median_buffer_range_percentage', 'max_slope', 'percent_amplitude',
                   'percent_difference_flux_percentile', 'pair_slope_trend_last_30', 'small_kurtosis', 'skew', 'std', 'magnitudeRatio', 'lombScargle', 'rcb']

ourFeatures = ['beyond1st',
               'kurtosis',
               'skew',
               'small_kurtosis',
               'std',
               'stetson_j',
               'stetson_k',
               'amplitude',
               'max_slope',
               'median_absolute_deviation',
               'median_buffer_range_percentage',
               'pair_slope_trend',
               'pair_slope_trend_last_30',
               'percent_amplitude',
               'percent_difference_flux_percentile',
               'flux_percentile_ratio_mid20',
               'flux_percentile_ratio_mid35',
               'flux_percentile_ratio_mid50',
               'flux_percentile_ratio_mid65',
               'flux_percentile_ratio_mid80',
               'poly1_t1',
               'poly2_t2',
               'poly2_t1',
               'poly3_t3',
               'poly3_t2',
               'poly3_t1',
               'poly4_t4',
               'poly4_t3',
               'poly4_t2',
               'poly4_t1']


masterDict = {'allFeatures': {'features': allFeatures,         'binary': {}, '8class': {}},
              #   'disantoFeatures': {'features': disantoFeatures, 'binary': {}, '8class': {}},
              #   'ourFeatures': {'features': ourFeatures,         'binary': {}, '8class': {}}
              }

rfHyperparams = {
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}


# svmHyperparams = {
#     'kernel': ['linear'],
#     'C': [0.125],
#     'gamma': [0.125]
# }

svmHyperparams = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': [2**x for x in range(-3, 6, 4)],
    'gamma': [2**x for x in range(-3, 6, 4)]
}


for problem in ['binary', '8class']:

    binTest, binTrain = splitTrainTest(problem)

    # the different feature sets
    for featureSet in masterDict.keys():

        classifiers = [
            # RFClassifier(binTrain, binTest, rfHyperparams,
            #              masterDict[featureSet]['features']),
            SVMClassifier(binTrain, binTest, svmHyperparams,
                          masterDict[featureSet]['features'])
        ]

        for classifier in classifiers:

            resultsPath = '../results/' + featureSet + '/' + \
                classifier.getClassifierType() + '/' + problem + '/'

            os.system('mkdir -p '+resultsPath)

            classifierPath = resultsPath+'classifier.pkl'

            if not path.exists(classifierPath):

                print('training...')
                classifier.train()
                print('Done training.')

                masterDict[featureSet][problem]['clf'] = classifier.getClassifier()

                print('saving classifier...')

                # save classifier
                with open(classifierPath, 'wb') as f:
                    pickle.dump(masterDict[featureSet][problem]['clf'], f)

            else:

                with open(classifierPath, 'rb') as f:
                    masterDict[featureSet][problem]['clf'] = pickle.load(f)
                classifier.trainedClasifier = masterDict[featureSet][problem]['clf']

            if classifier.getClassifierType() == 'RF':
                print('plotting importance plots...')
                plotFeatImportances(masterDict[featureSet][problem]['clf'],
                                    masterDict[featureSet]['features'], resultsPath + 'featureImportances.pdf')

            print('saving metrics and cm...')
            genMetricsAndCM(binTest,
                            classifier,
                            resultsPath + 'metrics.csv',
                            '../results/performances.csv',
                            problem,
                            featureSet)
