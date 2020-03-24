from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class RFClassifier:

    def __init__(self, trainData, testData, hyperparameters, features):
        self.trainData = trainData
        self.testData = testData
        self.hypermarameters = hyperparameters
        self.features = features
        self.trainedClasifier = None

    def getClassifierType(self):
        return 'RF'

    def train(self):

        # metrics to be analized

        def scorers():
            scoring = {'accuracy': make_scorer(accuracy_score),
                       'precision': make_scorer(precision_score, average='weighted'),
                       'recall': make_scorer(recall_score, average='weighted'),
                       'f1_score': make_scorer(f1_score, average='weighted')
                       }
            return scoring

        # learning
        model = RandomForestClassifier(random_state=0, class_weight='balanced')
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
        if self.trainedClasifier is not None:
            return self.trainedClasifier
        else:
            raise Exception('The model has not been trained yet!!')
