from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class SVMClassifier:

    def __init__(self, trainData, testData, hyperparameters, features):
        self.trainData = trainData
        self.testData = testData
        self.hypermarameters = hyperparameters
        self.features = features
        self.trainedClasifier = None
        self.trainDataMeans = None
        self.trainDataStds = None
        self.testNormalized = False

    def getClassifierType(self):
        return 'SVM'

    def checkModelIsTrained(self):
        if self.trainedClasifier is None:
            raise Exception('The model has not been trained yet!!')

    def calcTrainStats(self):
        self.trainDataMeans = self.trainData[self.features].mean()
        self.trainDataStds = self.trainData[self.features].std()

    def train(self):

        # normalize input
        # all columns are numeric values
        # only need to do standard normalization for every column
        self.calcTrainStats()
        self.trainData[self.features] = (
            self.trainData[self.features]-self.trainDataMeans)/self.trainDataStds

        # metrics to be analized

        def scorers():
            scoring = {'accuracy': make_scorer(accuracy_score),
                       'precision': make_scorer(precision_score, average='weighted'),
                       'recall': make_scorer(recall_score, average='weighted'),
                       'f1_score': make_scorer(f1_score, average='weighted')
                       }
            return scoring

        # learning
        model = SVC(random_state=0, class_weight='balanced')
        grid_search = GridSearchCV(model,
                                   self.hypermarameters,
                                   cv=StratifiedKFold(2),
                                   scoring=scorers(),
                                   refit='f1_score',
                                   return_train_score=True,
                                   verbose=100)

        # train the model
        grid_search.fit(self.trainData[self.features], self.trainData.Class)

        # Copy classifier
        self.trainedClasifier = grid_search

    def getClassifier(self):

        self.checkModelIsTrained()

        return self.trainedClasifier

    def predict(self):

        self.checkModelIsTrained()

        # in case the train data mean and std
        # hasnt been calculated yet
        if self.trainDataMeans is None:
            self.calcTrainStats()

        # normalize the test set once
        if not self.testNormalized:
            self.testData[self.features] = (
                self.testData[self.features]-self.trainDataMeans)/self.trainDataStds
            self.testNormalized = True

        return self.trainedClasifier.predict(self.testData[self.features])
