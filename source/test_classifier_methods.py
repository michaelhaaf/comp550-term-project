import classifier

from unittest import TestCase


class TestClassifierMethods(TestCase):
    def test_balance_test_sets(self):
        test_set_size = 2

        # 3x label1, 2x label2, 1x label3
        y_test = ['label1', 'label3', 'label1', 'label2', 'label1', 'label2']
        X_test = ['data1_1', 'data3_1', 'data1_2', 'data2_1', 'data1_3', 'data2_2']

        X_test_result, y_test_result = classifier.balance_test_sets(X_test, y_test, 2)

        # results: no label3 data (too small), only 2 of label1 (cut off any more than test_set_size)
        self.assertEqual(['data1_1', 'data1_2', 'data2_1', 'data2_2'], X_test_result)
        self.assertEqual(['label1', 'label1', 'label2', 'label2'], y_test_result)


    def test_remove_redundant_training_sets(self):
        y_test = ['label1', 'label1']
        y_train = ['label1', 'label2']
        X_train = ['data1_1', 'data2_1']

        X_train_result, y_train_result = classifier.remove_redundant_training_sets(y_test, y_train, X_train)

        self.assertEqual(y_train_result, ['label1'])
        self.assertEqual(X_train_result, ['data1_1'])
