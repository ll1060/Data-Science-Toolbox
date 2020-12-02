#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from random import shuffle


class Classifier:
    """Superclass of Classifier class
       Represents 3 different classfiers as subclasses
    """
    def __init__(self):
        """Constructor of the superclass
           member attibutes:
           train_set
           train_label
           test_set
           test_label

           methods:
           train
           test
        """
        self.train_set = None
        self.train_label = None
        self.test_set = None
        self.test_label = None

    def train(self, train_set, train_label):
        """set train_set and train_label
           assumes that the last column is always the label column
        """
        self.train_set = train_set
        self.train_label = train_label

    def test(self, test_set):
        """set test_set and pred_lable
        """
        self.test_set=test_set
        self.pred_label = None

class simpleKNNClassifier(Classifier):
    """Subclass of Classifier class
       Represents a simple KNN classifier algorithm
    """

    def __init__(self,k=3):
        """Constructor of the superclass
               member attibutes:
               k: int, number of nearsest neighbors
               train_set
               train_label
               test_set
               test_label

               methods:
               train
               test
        """
        super().__init__()
        self.k = k

    def train(self,train_set, train_label):
        """set train_set and train_label
           assumes that the last column is always the label column
        """
        super(simpleKNNClassifier,self).train(train_set, train_label)

    def _euclidean_distance(self,dp1,dp2):
        """calculate euclidean distance between pairs of data points
        """
        distance = 0.0
        for i in range(len(dp1)):
            dp1 = np.asarray(dp1)
            dp2 = np.asarray(dp2)

            distance += (dp1.astype(np.float)[i]-dp2.astype(np.float)[i])**2
        return np.sqrt(distance)

    def _predict(self, test_ins):
        """find values of target variable for k train_set rows with smallest distance
        """
        distances = []
        for i in range(len(self.train_set)):
            dist = self._euclidean_distance(self.train_set[i],test_ins)
            distances.append((self.train_label[i],dist))
        distances.sort(key=lambda x:x[1])

        #list to store k nearest neighbors
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])

        classes={}
        for i in range(len(neighbors)):
            response = neighbors[i]
            if response in classes:
                classes[response] += 1
            else:
                classes[response] = 1
        sorted_classes = sorted(classes.items(),key=lambda x:x[1],reverse=True)
        return sorted_classes[0][0]

    def test(self,test_set):
        """ make predictions on each element in the test_set
        """
        super(simpleKNNClassifier,self).test(test_set)
        pred_label = []
        for row in self.test_set:
            prediction = self._predict(row)
            pred_label.append(prediction)
        self.pred_label = pred_label
        return self.pred_label




# In[ ]:





# In[2]:




class Experiment:

    def __init__(self, dataset, true_labels, classifier_list):
        """This is the constructor of the Experiment class
           parameter:
           dataset: dataset to be used in the experiment
           true_label: list of true labels
           classifier_list: list of classifiers to be used on the dataset
           methods:
           _shuffle_data_and_label
           cv_split
           crossValidation
           Score
           _confusionMatrix
        """
        self.dataset = dataset
        self.true_labels = true_labels
        self.classifier_list = classifier_list


    def _shuffle_data_and_label(self):
        """shuffles the dataset and true_labels so that dataset[i] is still
           related to true_labels[i]
           prepare for cross validation
        """
        data_shuf = []
        label_shuf = []
        index_shuf = list(range(len(self.dataset)))
        shuffle(index_shuf)
        for i in index_shuf:
            data_shuf.append(self.dataset[i])
            label_shuf.append(self.true_labels[i])
        return data_shuf, label_shuf

    def cv_split(self, kFolds):
        """Split dataset into k folds
           parameter:
           dataset
           kFold: integer, number of groups that the given dataset to be split into
        """
        split_data = []
        split_label = []
        data_shuffled,label_shuffled = self._shuffle_data_and_label()
        fold_size = (len(data_shuffled))//kFolds

        for i in range(kFolds) :
            fold_data = []
            fold_label = []
            # since the data and label are both shuffled,
            # first fold will have data_shuf[0:fold_size]
            # second fold will have data_shuf[fold_size:2*fold_size]
            # so on ...
            for j in range(fold_size):
                fold_data.append(data_shuffled[i * fold_size + j])
                fold_label.append(label_shuffled[i * fold_size + j])

            split_data.append(fold_data)
            split_label.append(fold_label)

#         print("fold size::::::;", fold_size)
#         print(len(split_data), len(split_label))

#         for i in split_data:
#             print("fold size", len(i))

        return split_data, split_label




    def crossValidation(self, kFolds):
        """Performs cross validation for a dataset with a list of classifiers
           parameter:
           kFold: number of groups that the given dataset to be split into
        """
        numClassfiers = len(self.classifier_list)
        numSamples = len(self.dataset)

        #initialize the pred_label to be of size numSamples*numClassfiers
        pred_labels_for_all = []
        split_x, split_y = self.cv_split(kFolds)
        scores = []

        for this_classifier in self.classifier_list:
            pred_label_for_this_classifier = []
            if not isinstance(this_classifier,Classifier):
                raise Exception('The classifier is not applicable.')

            for run_fold in range(kFolds):

                test_x = split_x[run_fold]
                true_y = split_y[run_fold]

                # sum the rest of the lists in split_x and split y as training sets
                # sum with [] to flatten out the list
                train_x=[]
                train_y=[]

                if run_fold < kFolds-1:
                    train_x =split_x[:run_fold]+split_x[run_fold+1:]
                    train_y = split_y[:run_fold]+split_y[run_fold+1:]
                    ## flatten the list of lists
                    train_x = sum(train_x,[])
                    train_y = sum(train_y,[])
                else:
                    train_x = split_x[:run_fold]
                    train_y = split_y[:run_fold]
                    ## flatten the list of lists
                    train_x = sum(train_x,[])
                    train_y = sum(train_y,[])



#                 print("train x size: " , len(train_x) )
#                 print("train y size: " , len(train_y) )
#                 a = {}
#                 for i in train_y:
#                     if i in a:
#                         a[i] += 1
#                     else:
#                         a[i] = 1
#                 print("summary of train y: ", a)


                this_classifier.train(train_x,train_y)
                predicted_y = this_classifier.test(test_x)
                pred_label_for_this_classifier.append(predicted_y)
            pred_labels_for_all.append(pred_label_for_this_classifier)

        return split_y, pred_labels_for_all

    def score(self,true_label,predicted_label):
        """ Calculates accuracy of the classifier
            as percentage of number of correct predictions
        """
        accuracy_score = []
        correct=0

        for i in range(len(true_label)):
            if true_label[i]==predicted_label[i]:
                correct += 1
        accuracy = 100*(correct/(len(true_label)))
        accuracy_score.append(accuracy)

        return accuracy_score



    def confusionMatrix(self,true_label,predicted_label):
        """Computes a confusion matrix using numpy for two np.arrays

        """
        #get number of unique labels
        numClasses = len(set(true_label))
#         confusion_mat = [[0]*numClasses for i in range(numClasses)]
        sparse = False
        if sparse:
            confusion_mat = {}
            for pred, true in zip(predicted_label, true_label):
                if pred in confusion_mat:
                    if true in confusion_mat[pred]:
                        confusion_mat[pred][true] += 1
                    else:
                        confusion_mat[pred][true] = 1
                else:
                    confusion_mat[pred] = {}
                    confusion_mat[pred][true] = 1
            return confusion_mat
        else:
            all_label = set(predicted_label)

            confusion_mat_2d = [[0]*numClasses for i in range(numClasses)]
            label_encoder = dict((l, i) for i, l in enumerate(all_label))

            for pred, true in zip(predicted_label, true_label):

                pred_label_ecnoded = label_encoder[pred]
                true_label_encoded = label_encoder[true]
                confusion_mat_2d[pred_label_ecnoded][true_label_encoded] += 1
            print(label_encoder)
            plt.matshow(confusion_mat_2d)
            plt.show()
            return confusion_mat_2d
#         conf_mat = np.array( tuple(dict.values()) )


# In[ ]:
