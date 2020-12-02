#!/usr/bin/env python
# coding: utf-8

# In[13]:


from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from deliverable02 import DataSet
from deliverable03 import Classifier


# In[14]:


class ARMrule:
    """Superclass of transcation data
       It can generate the all ARM rule of dataset.
    """
    def __init__(self,data):
        """Constructor of the superclass
           member attibutes:
           my_data
           supportThreshold

           methods:
           element
           apriori
        """
        self.my_data=data
        self.supportThreshold=0.25

    def element(self):
        """
        Generate a list of subsetes which include only one element of data
        Parameter:
            header: bool, asks if the file has hearders


        """
        ele1=[]
        for i in self.my_data:
            for j in i:
                if not {j} in ele1:
                    ele1.append({j})
                    ele1.sort()
        return list(map(frozenset,ele1))

    def find_freq(self,ele):
        """
        Generate all the transaction and support of one element. And only keep the support value is greter than 0.25
        Parameter:
            ele: one element of data
        """
        trans={}
        for i in self.my_data:
            for j in ele:
                if j.issubset(i):
                    trans[j]=trans.get(j,0)+1
        n=float(len(self.my_data))
        freq_transaction=[]
        support_data={}
        for i in trans:
            support=trans[i]/n
            support_data[i]=support
            if support>=self.supportThreshold:
                freq_transaction.append(i)
        return support_data,freq_transaction

    def freqelement(self,transaction):
        '''
        Use a transaction to compare with other transaction. Then keep the subset which support value is greater than 0.25 as the after subset.
        Parameter:
            None
        '''
        ele=[]
        k=len(transaction)
        for i in range(k):
            for j in range(i+1,k):
                a=transaction[i]
                b=transaction[j]
                c=a|b
                if (not c in ele) and (len(c)==len(transaction[0])+1):
                    ele.append(c)
        return ele

    def apriori(self):
        """
        Main function to do apriori calculate. It will generate all the support value and all frequency function.
        Parameter:
            None
        """
        ele1=self.element()
        support_data,freq_transaction_1=self.find_freq(ele1)
        freq_transaction=[freq_transaction_1]
        k=2
        while len(freq_transaction[-1])>0 :
            ele=self.freqelement(freq_transaction[-1])
            support_data_k,freq_transaction_k=self.find_freq(ele)
            support_data.update(support_data_k)
            freq_transaction.append(freq_transaction_k)
            k+=1
        return support_data,freq_transaction

    def create_subset(self,old,new):
        """
        Add all element from old list to new list. Then generate a new subset.
        Parameter:
            Old: a list contain element which want to add to new list
            New: a list which want to receive the element from old list

        """
        for i in range(len(old)):
            t=[old[i]]
            tt=frozenset(set(old)-set(t))
            if not tt in new:
                new.append(tt)
                tt=list(t)
                if len(tt)>1:
                    self.create_subset(tt,new)
            return None

    def calculate(self,fre_set,all_set,support_data,rulelist):
        """
        Calculate all the confidence value and lift value of each transaction.
        Parameter:
            fre_set: the subset include one of transactions.
            _set: the set include all the fre_set
            port_data: the data include all the support value
            elist: a empty list which be used to store all the information,
	    """

        for after in all_set:
            conf=support_data[fre_set]/support_data[fre_set-after]
            lift=support_data[fre_set]/(support_data[fre_set-after] * support_data[after])
            a=(fre_set-after,'-->',after,':',
               'support:',round(support_data[fre_set],3),',',
               'conf:',round(conf,3),',',
               'lift:',round(lift,3))
            rulelist.append(a)
        return rulelist

    def __ARM__(self):
        """
        Use apriori function, create_subset function, and calculate function, to generate all the rules. And save all of them into a list.
        Parameter:
            None
        """

        support_data,freq_transaction=self.apriori()
        rulelist=[]
        for i in range(1,len(freq_transaction)):
            for fre_set in freq_transaction[i]:
                fre_list=list(fre_set)
                all_subset=[]
                self.create_subset(fre_list,all_subset)
                self.calculate(fre_set,all_subset,support_data,rulelist)
        return rulelist


# In[15]:


class TransactionDataSet(DataSet,ARMrule):

    def __init__(self,filename):
        """Constructor of the class
               member attibutes:
               filename

               methods:
               __readFromCSV
               __load
               clean
               explore
        """
        super().__init__(self)
        DataSet.__init__(self,filename)
        self.supportThreshold=0.25

    def __readFromCSV(self, filename, header=False):
        """ Read in the dataset that is in a CSV file

            Parameter:
            header: bool, asks if the file has hearders
        """
        # open the file and store rows in the file into a list
        # if fails to open the file, will prompt an error message
        try:
            with open(os.getcwd()+'/'+filename,'r') as file:
                self.my_reader = csv.reader(file, delimiter = ',')
                self.my_data = list(self.my_reader)
#                 print(len(self.my_data))
#                 print(self.my_data[1:3])
        except:
            print("This file is not a CSV file.\n")

    def __load(self, filename):
        """ check if filename is given and if file locates in current directory
            if both True, will call __readFromCSV

        """
        # if filename is not given,
        # prompt user and ask for filename

        if not filename:
                filename = input('Enter your filename to be loaded: ')

        # check if filename is right and if file is in current working directory
        # if True, will call __readFromCSV fucntion to load the filename
        # else will prompt an error message
        if os.path.exists(os.getcwd()+'/'+filename):
            self.__readFromCSV(filename)
        else:
            print('The file does not exist in the current working directory.')

    def clean(self):
        """
        a clean function to remove all the blank from the origin data.
        Parameter:
            None
        """
        newdata=[]
        for i in self.my_data:
             a= [j for j in i if j != '']
             newdata.append(a)
        self.my_data=newdata
        return self.my_data

    def explore(self):
        """
        Call the __ARM__ to get all the rules. Then get the top 10 rules by lift value.
        Parameter:
            None
        """
        rules=self.__ARM__()
        sup=[]
        index=list(range(0,len(rules)))
        top10_sup=[]
        for i in rules:
            sup.append(i[5])
        for i in range(0,10):
            i10=sup.index(max(sup))
            top10_sup.append(rules[index[i10]])
            sup.pop(i10)
            index.pop(i10)
        conf=[]
        index=list(range(0,len(rules)))
        top10_conf=[]
        for i in rules:
            conf.append(i[8])
        for i in range(0,10):
            i10=conf.index(max(conf))
            top10_conf.append(rules[index[i10]])
            conf.pop(i10)
            index.pop(i10)
        lift=[]
        index=list(range(0,len(rules)))
        top10_lift=[]
        for i in rules:
            lift.append(i[11])
        for i in range(0,10):
            i10=lift.index(max(lift))
            top10_lift.append(rules[index[i10]])
            lift.pop(i10)
            index.pop(i10)
        return top10_sup, top10_conf, top10_lift


# In[16]:


class Node:
    """A class where set a few attribute which will help to build the DecisionTree class.
       It can generate the node which will be used in DecisionTree class.
       member attributes:
           predicted
           index
           threshold
           left
           right
      Complexity analysis:
          Time: T(n)=n+1+1+1+1=n+4=O(n)
          Space: S(n)=n+1+1+0+0=n+2=O(n)
    """
    def __init__(self, predicted_class):
        self.predicted_class=predicted_class  #T(n)=n S(n)=n
        self.index=0 #T(n)=1 S(n)=1
        self.threshold=0 #T(n)=1 S(n)=1
        self.left=None #T(n)=1 S(n)=0
        self.right=None #T(n)=1 S(n)=0


# In[17]:


class DecisionTree:
    """Superclass of DecisionTreeClassifier.
       It can generate the desion tree of dataset.
    """
    def __init__(self,max_depth):
        """Constructor of the superclass
           member attibutes:
           max_depth

           methods:
           fit
           predict
        """
        self.max_depth=max_depth

    def split(self, traindata, trainlabel):
        """
        Find the best split for a node with best index and best threshold
        Parameters:
        traindata : the train data set.
        trainlabel : the real label of train data set.
        Complexity analysis:
            Time:
                best case:T(n)=1+1=2=O(1)
                worst case:T(n)=1+n+n+2+5n^3+n^2long+4n^2=5n^3+n^2long+4n^2+2n+3=O(n^3)
            Space:
                best case: S(n)=1+0=1=0(1).
                worst case:S(n)==1+n+n+0+5n^3+n^2long+4n^2=5n^3+n^2long+4n^2+2n+1=O(n^3)
        """
        m=trainlabel.size #T(n)=1 S(n)=1
        if m <= 1: #when m<=1, it will be the best case.
            return None, None #T(n)=1 S(n)=0
        parent = [np.sum(trainlabel == i) for i in range(self.classes)] #T(n)=n S(n)=n
        best_gini = 1.0 - sum((i / m) ** 2 for i in parent) #T(n)=n S(n)=n
        best_idx, best_thr = None, None #T(n)=2 S(n)=0
        for idx in range(self.features):
            # under this for loop, there zip sort functiom,and a copy function.
            # For time complexity of this entire for loop should be T(n)=n(nlogn+n+n+kn+5n)=(9+k)n^2+n^2long=O(nlogn)
            # For space complexity of this entire for loop should be S(n)=n(n+1+n+n+kn+5n)=(8+k)n^2+n^2long=O(nlogn)
            thresholds, classes = zip(*sorted(zip(traindata[:, idx], trainlabel))) #T(n)=nlogn S(n)=n+1
            left2 = [0] * self.classes #T(n)=n S(n)=n
            right2 = parent.copy() #T(n)=n S(n)=n
            for i in range(1, m):
                # under this for loop, there is two if condition
                # For time complexity of this entire for loop should be T(n)=n(1+1+1+1+1+k)=kn+5n K is number of fetures
                # For space complexity of this entire for loop should be S(n)=n(1+1+1+1+1+k)=kn+5n K is number of fetures
                x = classes[i - 1] #T(n)=1 S(n)=1
                left2[x] += 1  #T(n)=1 S(n)=1
                right2[x] -= 1 #T(n)=1 S(n)=1
                gini_left = 1.0 - sum((left2[j] / i) ** 2 for j in range(self.classes)) #T(n)=1 S(n)=1
                gini_right = 1.0 - sum((right2[j] / (m - i)) ** 2 for j in range(self.classes)) #T(n)=1 S(n)=1
                gini = (i * gini_left + (m - i) * gini_right) / m  #T(n)=1 S(n)=1
                if thresholds[i] == thresholds[i - 1]:
                    continue #log(n)
                if gini < best_gini:
                    best_gini = gini  #T(n)=1 S(n)=1
                    best_idx = idx #T(n)=1 S(n)=1
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2 #T(n)=1 S(n)=1
        return best_idx, best_thr

    def createtree(self, traindata, trainlabel, depth=0):
        """
        Use recursive function to create a decision tree.
        traindata : the train data set.
        trainlabel : the real label of train data set.
        Complexity analysis:
            Time: T(n)=n+n+n+d*n^3+n+n+1+1+1+1=d*n^2logn+5n+4=O(d*n^2logn) D is the max depth of tree.
            Space: S(n)=n+n+n+d*n^3+n+n+1+1+1+1=d*n^2logn+5n+4=O(d*n^2logn) D is the max depth of tree.
        """
        samples=[np.sum(trainlabel == i) for i in range(self.classes)] #T(n)= S(n)=n
        predicted_class = np.argmax(samples) #T(n)=n S(n)=n
        node = Node(predicted_class) # #T(n)=n S(n)=n
        if depth < self.max_depth:
            #T(n)=d*n^3  S(n)=d*n^3
            idx, threshold = self.split(traindata, trainlabel) # T(n)=n^2logn S(n)=n^2logn
            if idx is not None:
                indices_left = traindata[:, idx] < threshold #T(n)=n  S(n)=n
                traindata_left, trainlabel_left = traindata[indices_left], trainlabel[indices_left] #T(n)=n  S(n)=n
                traindata_right, trainlabel_right = traindata[~indices_left], trainlabel[~indices_left] #T(n)=n  S(n)=n
                node.index = idx #T(n)=1  S(n)=1
                node.threshold = threshold #T(n)=1  S(n)=1
                node.left = self.createtree(traindata_left, trainlabel_left, depth + 1) #T(n)=1  S(n)=1
                node.right = self.createtree(traindata_right, trainlabel_right, depth + 1) #T(n)=1  S(n)=1
        return node

    def fit(self,traindata,trainlabel):
        """
        Create decision tree classifier.
        Complexity analysis:
            Time: T(n)=1+1+n^3=n^3+2=O(n^3)
            Space: S(n)=n+n+n+d*n^3+n+n+1+1+1+1=d*n^2logn+5n+4=O(d*n^2logn) D is the max depth of tree.
        """
        self.classes=len(set(trainlabel)) #T(n)=1 S(n)=1
        self.features=traindata.shape[1]  #T(n)=1 S(n)=1
        self.tree= self.createtree(traindata,trainlabel) #T(n)=d*n^2logn S(n)=d*n^2logn D is the max depth of tree.

    def predict(self,testdata):
        """
        Predict the test data and generate predicted label.
        Paremeter:
            testdata: the dataset which used to generate predict label
        Complexity analysis:
            Time: T(n)=m+d*logn=O(d*logn) D is the max depth of tree. M is the number of features of tree.
            Space: S(n)=T(n)=m+d*logn=O(d*logn) D is the max depth of tree. M is the number of features of tree.
        """
        node = self.tree #T(n)=m S(n)=m M is the number of features of tree.
        while node.left:
            #Under while this is a if-else:
            #For time, T(n)=d*log(n)  D is the max depth of tree
            #For space, S(n)=d*log(n)  D is the max depth of tree
            if testdata[node.index] < node.threshold:
                node = node.left #T(n)=1 S(n)=1
            else:
                node = node.right #T(n)=1 S(n)=1
        return node.predicted_class



# In[ ]:





# In[18]:




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
        self.pred_prob = None

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

        return sorted_classes[0][0], sorted_classes[0][1]/self.k

    def test(self,test_set):
        """ make predictions on each element in the test_set
        """
        super(simpleKNNClassifier,self).test(test_set)
        pred_label = []
        pred_prob = []
        for row in self.test_set:
            prediction, pred_prob_ins = self._predict(row)
            pred_label.append(prediction)
            pred_prob.append(pred_prob_ins)
        self.pred_label = pred_label
        return self.pred_label, pred_prob


class DecisionTreeClassifier(Classifier,DecisionTree):
    """Subclass of Classifier class
       Represents a decision tree classifier algorithm
    """
    def __init__(self,max_depth,data_type='quantitive'):
        """Constructor of the class
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
        Classifier.__init__(self)
        self.max_depth=max_depth
        self.data_type=data_type

    def transdata(self,data):
        """
        A function transform the variabel of qualitative data into 0,1,2,3... which is easier to do the decision tree classifier.
        data : qualitative data
        Complexity analysis:
            Time: T(n)=n+n^3+n^2+n+2=n^3+n^2+2n+2=O(n^3)
            Space: S(n)=n+n(m+n)=n^2+mn+n=O(n^2). m is number of categories of each variable.
        """
        newdata=np.empty(data.shape, dtype=int) #T(n)=n S(n)=0
        for i in range(data.shape[1]):
            #under this for loop there is two for loop
            #T(n)=1+1+n(n^2)+n)+n=n^3+n^2+n+2
            #S(n)=m+n
            new1=[] #T(n)=1 S(n)=0
            new2=[] #T(n)=1 S(n)=0
            for j in data[:,i]:
            # under this for loop, there is a if condition and a append function.
            # For time complexity of this entire for loop should be T(n)=n*n=n^2
            # For space complexity of this entire for loop should be S(n)=m. m is number of catogories of the variable.
                if j not in new1:
                    new1.append(j)
            for j in data[:,i]:
            #under this for loop, there is a append function. In append function there is a index function.
            #T(n)=n*1*1=n
            #S(n)=n
                new2.append(new1.index(j))
            newdata[:,i]=new2  ##T(n)=n S(n)=n
        return newdata

    def train(self, traindata, trainlabel):
        """
        Transform train label to be in numeric in to (0,1,2,3.....) as new label which is easier to do decision tree classifier
        set traindaata, real train label, and new label.
        Paremeter:
            traindata : the train data set.
            trainlabel : the real label of train data set.
        Complexity analysis:
            Time:
                best case:T(n)=1+1+n^2+n+n+n+n=n^2+4n+2=O(n^2)
                worst case:T(n)1+1+n^2+n+n+n+n^3=n^3+n^2+3n+2=o(n^3)
            Space:
                best case: S(n)=0+0+k+n+n+n+n=4n+k=O(n) K is number of categories of label.
                worst case:S(n)=0+0+k+n+n+n+n^2=n^2+3n+k=O(n) K is number of categoreis of label.
        """
        self.element=[]   #T(n)=1 S(n)=0
        newlabel=[]       #T(n)=1 S(n)=0
        for i in trainlabel:
            # under this for loop, there is a if condition and a append function.
            # For time complexity of this entire for loop should be T(n)=n*n=n^2
            # For space complexity of this entire for loop should be S(n)=k. K is number of catogories of the label.
            if i not in self.element:
                self.element.append(i)
        for i in trainlabel:
            #under this for loop, there is a append function. In append function there is a index function.
            #T(n)=n*1*1=n
            #S(n)=n
            newlabel.append(self.element.index(i))
        self.newlabel=np.array(newlabel) #T(n)=n S(n)=n
        self.reallabel=trainlabel #T(n)=n S(n)=n
        if self.data_type=='quantitive':
            self.traindata=traindata #T(n)=n S(n)=n
            return self.traindata,self.newlabel,self.reallabel
        elif self.data_type=='qualitative':
            self.traindata=self.transdata(traindata) #T(n)=n^3 S(n)=n^2
            return self.traindata,self.newlabel,self.reallabel
        else:
            print('Decisicon tree can only accept quantitive data or qualitive data.')


    def test(self,testdata):
        """
        create predicted label for each element in the testdata
        Paremeter:
            testdata: the dataset which used to generate predict label
        Complexity analysis:
            Time:
                best case:T(n)=n+d*n^2logn+1+d*nlogn+1+n^2+n+n^2=d*n^2logn+d*nlogn_n^2+n^2+2n+2=O(n^2logn) D is the max depth of tree.
                worst case:T(n)=n+n^3+1+d*nlogn+1+n^2+n+n^2=n^3+d*nlogn_n^2+n^2+2n+2=O(n^3) D is the max depth of tree.
            Space:
                best case: S(n)=n+d*n^2logn+0+d*nlogn+0+n^2+n+n^2=d*n^2logn+d*nlogn_n^2+n^2+2n=O(n^2logn) D is the max depth of tree.
                worst case:S(n)=n+n^3+0+d*nlogn+0+n^2+n+n^2=n^3+d*nlogn_n^2+n^2+2n=O(n^3) D is the max depth of tree.
        """
        if self.data_type=='quantitive':
            #if data type is quantitive data, then this will be the best case.
            testdata=testdata   #T(n)=n S(n)=n
        elif self.data_type=='qualitative':
            testdata=self.transdata(testdata)  #T(n)=n^3 S(n)=n^2
        self.fit(self.traindata, self.newlabel) #T(n)=d*n^2logn S(n)=d*n^2logn  D is the max depth of tree.
        test_label=[]  #T(n)=1 S(n)=0
        for i in testdata:
            #T(n)=d*nlogn S(n)=d*nlogn
            test_label.append(self.predict(i)) #T(n)=d*logn S(n)=d*logn
        result=[]  #T(n)=1 S(n)=0
        for i in range(len(testdata)):
            #T(n)=n^2+n S(n)=n^2+n
            b=list(testdata[i])+[self.element[test_label[i]]] #T(n)=n S(n)=n
            result.append(b) #T(n)=1 S(n)=1
        pred_label=[]
        for i in range(len(test_label)):
            #T(n)=n S(n)=n
            pred_label.append(self.element[test_label[i]]) #T(n)=1 S(n)=1
        return pred_label,result

    def __str__(self):
        """
        Print the decision tree into str type.
        """
        tree=self.toString(self.tree)
        a=list(range(len(self.element)))
        for i in a:
            tree=tree.replace(str(i), self.element[i])
        return tree


    def toString(self,tree):
        """
        Convert decision tree to string type which are readable.
        Paremeter:
            tree: decison tree
        """
        result=''
        def recurse(tree,result):
            if tree==None:
                return str('None')
            if tree.left==tree.right==None:
                return str(tree.predicted_class)
            l=str(tree.predicted_class)
            if tree.left != None:
                l=l+'['+recurse(tree.left, result)+']'
            else:
                l=l+'[None]'
            if tree.right != None:
                l=l+'['+recurse(tree.right, result)+']'
            return l
        #return the result of recurse function.
        return '['+recurse(tree, result)





# In[ ]:





# In[19]:




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
        scores = [] # not in use
        pred_prob = []


        for this_classifier in self.classifier_list:
            pred_label_for_this_classifier = []
            pred_prob_for_this_classifier = []
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
                predicted_y, pred_prob_for_this_fold = this_classifier.test(test_x)
                pred_label_for_this_classifier.append(predicted_y)
                pred_prob_for_this_classifier.append(pred_prob_for_this_fold)
            pred_labels_for_all.append(pred_label_for_this_classifier)
            pred_prob.append(pred_prob_for_this_classifier)

        return split_y, pred_labels_for_all, pred_prob

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

    def ROC_curve(self,true_label,predicted_label,pred_prob):
        """
        Generates a plot for ROC curves.

        Complexity analysis:(Assume worst case)
            c = number of unique classes, n = number of test samples
            Time: T(n)= 2c + 2n + 20 + c*103*(10+7n) =1046c+723*c*n + 2n = O(n) 
            Space: S(n)= 2+103c(10+n) = 1030c+103c*n+2 = O(n)
        """
        #get class names and counts of unique classes
        classification_classes = np.unique(true_label) # T(n)=n(number of test samples), S(n)=1
        unique_counts = len(set(true_label)) # T(n)=n+3, S(n)=1
        if unique_counts == 2: # T(n)=3,S(n)=0
            tpr=[] # T(n)=1,S(n)=1
            fpr=[] # T(n)=1,S(n)=1
            thresholds = np.arange(-0.01, 1.02, .01) # T(n)=1,S(n)=1
            for thresh in thresholds: # np.arange(-0.01, 1.02, .01) gives a total of 103 loops
                true_positive= np.nextafter(0,1) # T(n)=1,S(n)=1
                false_positive= np.nextafter(0,1) # T(n)=1,S(n)=1
                true_negative= np.nextafter(0,1) # T(n)=1,S(n)=1
                false_negative= np.nextafter(0,1) # T(n)=1,S(n)=1
                for i in range(len(predicted_label)): # n loops
                    if true_label[i]==predicted_label[i]: # T(n)=3,S(n)=0
                        if pred_prob[i]>=thresh: #T(n)=3,S(n)=0
                            true_positive+=1 #T(n)=1,S(n)=1
                        else:
                            false_negative+=1 #T(n)=1,S(n)=1
                    else:
                        if pred_prob[i]>=thresh: #T(n)=3,S(n)=0
                            false_positive+=1 #T(n)=1,S(n)=1
                        else:
                            true_negative+=1 #T(n)=1,S(n)=1
                tpr_t = true_positive/(true_positive + false_negative) #T(n)=2,S(n)=1
                fpr_t = false_positive/(false_positive + true_negative) #T(n)=2,S(n)=1
                tpr.append(tpr_t) #T(n)=1,S(n)=1
                fpr.append(tpr_t) #T(n)=1,S(n)=1
    #             print(thresh)
    #             print(true_positive, true_negative, false_positive, false_negative)
    #             print(tpr_t, fpr_t)

    #         print(tpr, fpr)
            plt.plot(tpr,fpr) #T(n)=1,S(n)=0
            plt.xlim(-0.01, 1.1) #T(n)=1,S(n)=0
            plt.ylim(-0.01, 1.1) #T(n)=1,S(n)=0
            plt.title('ROC curve') #T(n)=1,S(n)=0
            plt.show() #T(n)=1,S(n)=0

        elif unique_counts > 2:
            tpr_all = [] #T(n)=1,S(n)=1
            fpr_all = [] #T(n)=1,S(n)=1
            for each_class in classification_classes: # c loops (c=number of unique classes)
                tpr_this_class=[] #T(n)=1,S(n)=1
                fpr_this_class=[] #T(n)=1,S(n)=1
                thresholds = np.arange(-0.01, 1.02, .01) # np.arange(-0.01, 1.02, .01) gives a total of 103 loops
                for thresh in thresholds: # 103 loops
                    true_positive= np.nextafter(0,1)  # T(n)=1,S(n)=1
                    false_positive= np.nextafter(0,1)  # T(n)=1,S(n)=1
                    true_negative= np.nextafter(0,1)  # T(n)=1,S(n)=1
                    false_negative= np.nextafter(0,1)  # T(n)=1,S(n)=1
                    for i in range(len(predicted_label)): # n loops
                        if true_label[i]==each_class: #T(n)=3,S(n)=0
                            if pred_prob[i]>=thresh: #T(n)=3,S(n)=0
                                true_positive+=1 # T(n)=1,S(n)=1
                            else:                #T(n)=3,S(n)=0
                                false_negative+=1 # T(n)=1,S(n)=1
                        else:                     #T(n)=3,S(n)=0
                            if pred_prob[i]>=thresh: #T(n)=3,S(n)=0
                                false_positive+=1  # T(n)=1,S(n)=1
                            else:                  #T(n)=3,S(n)=0
                                true_negative+=1  # T(n)=1,S(n)=1
                    tpr_t = true_positive/(true_positive + false_negative)  #T(n)=2,S(n)=1
                    fpr_t = false_positive/(false_positive + true_negative)  #T(n)=2,S(n)=1
                    tpr_this_class.append(tpr_t) #T(n)=1,S(n)=1
                    fpr_this_class.append(fpr_t) #T(n)=1,S(n)=1
                    tpr_all.append(tpr_this_class) #T(n)=1,S(n)=1
                    fpr_all.append(fpr_this_class) #T(n)=1,S(n)=1
            plt.figure()   #T(n)=1,S(n)=0
            for i in range(len(tpr_all)):  # c loops
                plt.plot(tpr_all[i],fpr_all[i]) #T(n)=1,S(n)=0
                plt.legend(classification_classes) #T(n)=1,S(n)=0
            plt.xlim(-0.01, 1.1) #T(n)=1,S(n)=0
            plt.ylim(-0.01, 1.1) #T(n)=1,S(n)=0
            plt.title('ROC curve') #T(n)=1,S(n)=0
            plt.show() #T(n)=1,S(n)=0

        else:
            print('Not enough number of classess included to generate ROC curve.')
#         print(max(pred_prob))
#         print(tpf,fpf)




# #         return tpr,fpr;



# In[ ]:





# In[20]:
# from deliverable04 import TransactionDataSet,DecisionTreeClassifier
# from deliverable02 import QuantDataSet, QualitDataSet
#
# import numpy as np
#
# trans=TransactionDataSet('grocery.csv')
# trans._TransactionDataSet__readFromCSV('grocery.csv')
# data=trans.clean()
# #print(data)
# top10_sup,top10_conf,top10_list=trans.explore()
# print(top10_sup)
# print(top10_conf)
# print(top10_list)
#
#
# ##Quantitive decision tree
# quant=QuantDataSet('iris.csv')
# quant._QuantDataSet__load('iris.csv')
# quantdata=quant.my_data
#
# quantdata=np.array(quantdata)
# data = quantdata[:, :-1]
# data=data.astype(np.float)
# reallabel = quantdata[:,-1]
#
# tree=DecisionTreeClassifier(3)
# data1,reallabel1,newlabel1=tree.train(data,reallabel)
# data_predictlabel=tree.test(data)
# #print(data_predictlabel)
# print(tree)
#
# #qualitative' decision tree
# qual=QualitDataSet('new_iris.csv')
# qual._QualitDataSet__load('new_iris.csv')
# qualdata=qual.my_data
#
# qualdata=np.array(qualdata)
# data2=qualdata[1:,:-1]
# reallabel2 = quantdata[0:,-1]
# tree2=DecisionTreeClassifier(3,'qualitative')
# new_data,reallabel2,newlabel2=tree2.train(data2,reallabel2)
# data_predictlabel2=tree2.test(data2)
# print(data_predictlabel)
# print(tree2)
#
# a=tree.tree
#
#




# In[21]:
