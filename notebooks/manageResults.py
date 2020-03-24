import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def shorten(x):
    return {
        'stetson_j': 'stetson_j',
        'std': 'std',
        'median_absolute_deviation': 'mad',
        'amplitude': 'amp',
        'poly1_t1': 'poly1_t1',
        'poly2_t1': 'poly2_t1',
        'skew': 'skew',
        'poly3_t1': 'poly3_t1',
        'small_kurtosis': 'sk',
        'stetson_k': 'stetson_k',
        'median_buffer_range_percentage': 'mbrp',
        'percent_amplitude': 'p_amp',
        'percent_difference_flux_percentile': 'pdfp',
        'poly4_t1': 'poly4_t1',
        'poly3_t2': 'poly3_t2',
        'poly4_t2': 'poly4_t2',
        'max_slope': 'max_slope',
        'kurtosis': 'kurtosis',
        'pair_slope_trend': 'pst',
        'poly2_t2': 'poly2_t2',
        'beyond1st': 'beyond1st',
        'flux_percentile_ratio_mid35': 'fpr35',
        'flux_percentile_ratio_mid50': 'fpr50',
        'flux_percentile_ratio_mid65': 'fpr65',
        'flux_percentile_ratio_mid20': 'fpr20',
        'flux_percentile_ratio_mid80': 'fpr80',
        'pair_slope_trend_last_30': 'pst_last30',
        'poly3_t3': 'poly3_t3',
        'poly4_t3': 'poly4_t3',
        'poly4_t4': 'poly4_t4',
        'magnitudeRatio': 'mr',
        'lombScargle': 'ls',
        'rcb': 'rcb'
    }[x]


def plotFeatImportances(clf, features, savePath):
    # calculate feature importance in descending order
    importances = clf.best_estimator_.feature_importances_*100
    featsCopy = features.copy()

    Y = list(importances)
    X = list(featsCopy)

    yx = list(zip(Y, X))
    yx.sort()

    yx = yx[::-1]

    x_sorted = [x for y, x in yx]
    y_sorted = [y for y, x in yx]

    for i, x in enumerate(x_sorted):
        x_sorted[i] = shorten(x)

    print(len(x_sorted))
    print(len(y_sorted))
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(14, 8))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.bar(np.arange(len(y_sorted)), y_sorted,
            edgecolor='black', color='firebrick')
    plt.xticks(np.arange(len(y_sorted)), x_sorted, rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Feature importance(%)')
    plt.title("Binary Classification")
    plt.savefig(savePath)
    plt.close()


def genMetricsAndCM(testData, clf, features, completeSavePath, allResultsFilePath, problemType, classifierName, featureName):
    confMatr = confusion_matrix(testData.Class, clf.predict(
        testData[features])).transpose()

    def fMeasure(precision, recall):
        return 2*precision*recall/(precision+recall)

    file = open(completeSavePath, 'w')
    file.write('Precision,Recall,F-score,Cover\n')

    newMatr = []
    for i in range(len(confMatr)):
        prec = confMatr[i][i]/(sum(confMatr[i, :]))
        rec = confMatr[i][i]/(sum(confMatr[:, i]))
        newMatr.append([prec, rec, fMeasure(prec, rec), sum(confMatr[:, i])])
        file.write(str(prec)+',' +
                   str(rec) + ',' +
                   str(fMeasure(prec, rec)) + ',' +
                   str(sum(confMatr[:, i])) +
                   '\n')
    file.close()

    newMatr = np.array(newMatr)

    avePrec = np.average(newMatr[:, 0])
    aveRec = np.average(newMatr[:, 1])
    aveF1 = np.average(newMatr[:, 2])

    file = open(allResultsFilePath, 'a')
    # problem, classifier, features, prec, rec, f1
    file.write('{},{},{},{},{},{}\n'.format(
        problemType, classifierName, featureName, avePrec, aveRec, aveF1))

    file.close()
