import unittest
from splitTrainTest import splitTrainTest


class TestSplitTrainTest(unittest.TestCase):

    def test_binReplicability(self):
        test1, train1 = splitTrainTest('binary')
        test2, train2 = splitTrainTest('binary')
        assert (test1.equals(test2)) and (train1.equals(train2))

    def test_binSameNumberOfLcsPerClass(self):
        test1, train1 = splitTrainTest('binary')

        labels = train1.Class.unique()
        numExamplesInTrainPerClass = len(train1[train1.Class == labels[0]])

        for l in train1.Class.unique():
            assert len(train1[train1.Class == l]) == numExamplesInTrainPerClass, '''minNum {} label {} had {} '''.format(
                numExamplesInTrainPerClass, l, len(train1[train1.Class == l]))

    def test_binTraInNotInTestAndViceVersa(self):
        test1, train1 = splitTrainTest('binary')

        trainIds = train1.index.get_level_values('ID')
        testIds = test1.index.get_level_values('ID')

        assert len(train1[train1.index.get_level_values('ID').isin(testIds)]) == 0
        assert len(test1[test1.index.get_level_values('ID').isin(trainIds)]) == 0

    def test_8ClassReplicability(self):
        test1, train1 = splitTrainTest('8class')
        test2, train2 = splitTrainTest('8class')
        assert (test1.equals(test2)) and (train1.equals(train2))

    def test_8classSameNumberOfLcsPerClass(self):
        test1, train1 = splitTrainTest('8class')

        labels = train1.Class.unique()
        numExamplesInTrainPerClass = len(train1[train1.Class == labels[0]])

        for l in train1.Class.unique():
            assert len(train1[train1.Class == l]) == numExamplesInTrainPerClass, '''minNum {} label {} had {} '''.format(
                numExamplesInTrainPerClass, l, len(train1[train1.Class == l]))

    def test_8classTraInNotInTestAndViceVersa(self):
        test1, train1 = splitTrainTest('8class')

        trainIds = train1.index.get_level_values('ID')
        testIds = test1.index.get_level_values('ID')

        assert len(train1[train1.index.get_level_values('ID').isin(testIds)]) == 0
        assert len(test1[test1.index.get_level_values('ID').isin(trainIds)]) == 0



if __name__ == '__main__':
    unittest.main()
