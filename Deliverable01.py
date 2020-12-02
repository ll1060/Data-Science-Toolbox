#* Deliverable.py #*
#* ANLY 555 Fall 2020
#* Project Deliverable 1
#*
#*  Due on: 09/27/2020
#*  Author: Leyao Li
#*
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither
#* given nor received any assistance on this project other than
#* the TAs, professor, textbook and teammates. #*


#-------------- Class DataSet --------------#

class DataSet:
    """ Represent basic information for a two-dimensional dataset in tabular form
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file
        __load: a more generic method for reading in the dataset from a file
        clean: handles N/A values
        explore: show summary statistics of the dataset
        randomize: disorder rows in the dataset
    """


    def __init__(self, filename=None ):
        """ Create an instance of a DataSet

            Attributes:
            filename: path to where the file is located. The DataSet instance
            will be populated using the given file.
            If the filename is not given, an empty dataset will be created.

        """

        self.filename = filename
        print('inistantiate DataSet class')

    def __readFromCSV(self, filename, header=True):
        """ Read in the dataset that is in a CSV file
            The DataSet instance will be populated using the given file.


            Parameter:
            header: bool, asks if the file has hearders
        """


        print('readFromCSV method')

    def __load(self, filename):
        """ The generic method for reading in the dataset
            will determine file format and load data based on the format.

        """



        print('load method')

    def clean(self):
        """ handle N/A values in the dataset

            by either
            removing N/A values in a column or in a row with col/row names/index
            or
            fillING in with mean or interpolation

        """
        print ('clean method')


    def explore(self):
        """ show summary statistics of the dataset
        """

        print('explore method')


    def randomize(self):
        """ disorder all rows in the dataset, prepare for tests and cross validation
        """

        print('randomize method')


#-------------- Class TimeSeriesDataSet  --------------#

class TimeSeriesDataSet(DataSet):
    """ Represent a two-dimensional dataset indexed in time order
        Inherit from the superclass DataSet
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file
        __load: a more generic method for reading in the dataset from a file
        clean: handles N/A values
        explore: show summary statistics of the dataset
    """

    def __init__(self, filename=None ):
        """ Create an instance of a TimeSeriesDataSet



            Attributes:
            filename: path to where the file is located. The DataSet instance
            will be populated using the given file.
            If the filename is not given, an empty dataset will be created.

        """
        super().__init__(self)

    def __readFromCSV(self, filename, header=True):
        """ Read in the dataset that is in a CSV file
            The DataSet instance will be populated using the given file.


            Parameter:
            header: bool, asks if the file has hearders
        """


        return 'readFromCSV method'

    def __load(self, filename):
        """ The generic method for reading in the dataset
            will determine file format and load data based on the format.

        """



        return 'load method'

    def clean(self):
        """ handle N/A values in the dataset

            by either
            removing N/A values in a column or in a row with col/row names/index
            or
            fillING in with mean or interpolation

        """
        print('time series clean method')


    def explore(self):
        """ show summary statistics of the dataset
        """

        print('time series explore method')

#-------------- Class QuantDataSet --------------#

class QuantDataSet(DataSet):
    """ Represent a two-dimensional dataset with quantitative values
        Inherit from the superclass DataSet
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file
        __load: a more generic method for reading in the dataset from a file
        clean: handles N/A values
        explore: show summary statistics of the dataset
        randomize: disorder rows in the dataset
    """

    def __init__(self, filename=None ):
        """ Create an instance of a QuantDataSet



            Attributes:
            filename: path to where the file is located. The DataSet instance
            will be populated using the given file.
            If the filename is not given, an empty dataset will be created.

        """
        super().__init__(self)

    def __readFromCSV(self, filename, header=True):
        """ Read in the dataset that is in a CSV file
            The DataSet instance will be populated using the given file.


            Parameter:
            header: bool, asks if the file has hearders
        """


        return 'readFromCSV method'

    def __load(self, filename):
        """ The generic method for reading in the dataset
            will determine file format and load data based on the format.

        """



        return 'load method'

    def clean(self):
        """ handle N/A values in the dataset

            by either
            removing N/A values in a column or in a row with col/row names/index
            or
            fillING in with mean or interpolation

        """
        print('quant data clean method')


    def explore(self):
        """ show summary statistics of the dataset
        """

        print ('quant data explore method')

    def randomize(self):
        """ disorder all rows in the dataset, prepare for tests and cross validation
        """

        print('quant data randomize method')


#-------------- Class QualDataSet --------------#

class QualDataSet(DataSet):
    """ Represent a two-dimensional qualitative dataset
        Inherit from the superclass DataSet
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file
        __load: a more generic method for reading in the dataset from a file
        clean: handles N/A values
        explore: show summary statistics of the dataset
        randomize: disorder rows in the dataset
    """

    def __init__(self, filename=None ):
        """ Create an instance of a QualDataSet



            Attributes:
            filename: path to where the file is located. The DataSet instance
            will be populated using the given file.
            If the filename is not given, an empty dataset will be created.

        """
        super().__init__(self)

    def __readFromCSV(self, filename, header=True):
        """ Read in the dataset that is in a CSV file
            The DataSet instance will be populated using the given file.


            Parameter:
            header: bool, asks if the file has hearders
        """


        return 'readFromCSV method'

    def __load(self, filename):
        """ The generic method for reading in the dataset
            will determine file format and load data based on the format.

        """



        return 'load method'

    def clean(self):
        """ handle N/A values in the dataset

            by either
            removing N/A values in a column or in a row with col/row names/index
            or
            fillING in with mean or interpolation

        """
        print('qualitative data clean method')


    def explore(self):
        """ show summary statistics of the dataset

        """

        print('qualitative data explore method')


    def randomize(self):
        """ disorder all rows in the dataset, prepare for tests
            and cross validation
        """

        print('qualitative data randomize method')

#-------------- Class TextDataSet --------------#

class TextDataSet(DataSet):
    """ Represent a two-dimensional text dataset
        Inherit from the superclass DataSet
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file
        __load: a more generic method for reading in the dataset from a file
        clean: handles N/A values
        explore: show summary statistics of the dataset
        randomize: disorder rows in the dataset
    """

    def __init__(self, filename=None ):
        """ Create an instance of a QualDataSet



            Attributes:
            filename: path to where the file is located. The DataSet instance
            will be populated using the given file.
            If the filename is not given, an empty dataset will be created.

        """
        super().__init__(self)

    def __readFromCSV(self, filename, header=True):
        """ Read in the dataset that is in a CSV file
            The DataSet instance will be populated using the given file.


            Parameter:
            header: bool, asks if the file has hearders
        """


        return 'readFromCSV method'

    def __load(self, filename):
        """ The generic method for reading in the dataset
            will determine file format and load data based on the format.

        """



        return 'load method'

    def clean(self):
        """ handle N/A values in the dataset

            by either
            removing N/A values in a column or in a row with col/row names/index
            or
            filling in with mean or interpolation.

        """
        print('text data clean method')


    def explore(self):
        """ show summary statistics of the dataset.

        """

        print('text data explore method')

    def randomize(self):
        """ disorder all rows in the dataset, prepare for tests
            and cross validation
        """

        print('text data randomize method')


#-------------- Class ClassifierAlgorithm --------------#
class ClassifierAlgorithm:
    """ Represents a collection of classfication methods

        Methods:
        __init__: constructor of the method
        train: train the model on a training dataset
        test: test the model on a testing dataset

    """
    def __init__(self,data=''):
        """ Create an instance of the classifier algorithm"""
        self.data = data

        print('constructor of ClassifierAlgorithm')


    def train(self,testing_data):
        """ train the classification model with training dataset
        """

        print('train method')


    def test(self,training_data):
        """ test the classification model with testing dataset
        """

        print('test method')



#-------------- Class simpleKNNClassifier --------------#
class simpleKNNClassifier(ClassifierAlgorithm):
    """ Represents a simple KNN classfication method
        Inherit from the super class ClassifierAlgorithm

        Methods:
        __init__: constructor of the method
        train: train the model on a training dataset
        test: test the model on a testing dataset

    """
    def __init__(self, n=5, metric=''):
        """ Create an instance of a simple KNN Classifier

            Attributes:
            n: int, number of neighbors, default is 5
            metric: str, distance metric, default is 'minkowski'
        """
        super().__init__(self)
        self.n = 5
        self.metric='minkowski'

        print('constructor of simple KNN classifier')

    def train(self,training_data):
        """ train the classification model with training dataset

            Attributes:
            trainig_data: dataset used for training the classifier
        """

        print('train method of simple KNN classifier')


    def test(self, testing_data):
        """ test the classification model with testing dataset
            Attributes:
            testing_data: dataset used for testing the classifier
        """

        print('test method of simple KNN')




#-------------- Class kdTreeKNNClassifier --------------#
class kdTreeKNNClassifier(ClassifierAlgorithm):
    """ Represents a kdTree KNN classfication method
        Inherit from the super class ClassiferAlgorithm

        Methods:
        __init__: constructor of the method
        train: train the model on a training dataset
        test: test the model on a testing dataset

    """
    def __init__(self, n=5):
        """ Create an instance of a kd tree Classifier

            Attributes:
            n: int, number of neighbors, default is 5

        """
        super().__init__(self)
        self.n = 5

        print('constructor of kd Tree KNN Classifier')


    def train(self,training_data):
        """ train the classification model with training dataset

            Attributes:
            trainig_data: dataset used for training the classifier
        """

        print('train method of kd Tree Classifier')


    def test(self, testing_data):
        """ test the classification model with testing dataset
            Attributes:
            testing_data: dataset used for testing the classifier
        """

        print('test method of kd Tree Classifier')



#-------------- Class hmmClassifier --------------#
class hmmClassifier(ClassifierAlgorithm):
    """ Represents a hmm classfication method
        Inherit from the super class ClassifierAlgorithm

        Methods:
        __init__: constructor of the method
        train: train the model on a training dataset
        test: test the model on a testing dataset

    """
    def __init__(self):
        """ Create an instance of a hmm Classifier



        """
        super().__init__(self)

        print('constructor of hmm Classifier')

    def train(self,training_data):
        """ train the classification model with training dataset

            Attributes:
            trainig_data: dataset used for training the classifier
        """

        print('train method of hmm Classifier')


    def test(self, testing_data):
        """ test the classification model with testing dataset
            Attributes:
            testing_data: dataset used for testing the classifier
        """

        print('train method of hmm Classifier')

#-------------- Class graphKNNClassifer --------------#
class graphKNNClassifier(ClassifierAlgorithm):
    """ Represents a graph KNN classfication method
        Inherit from the super class ClassifierAlgorithm

        Methods:
        __init__: constructor of the method
        train: train the model on a training dataset
        test: test the model on a testing dataset

    """
    def __init__(self):
        """ Create an instance of a graph KNN Classifier



        """
        super().__init__(self)


        print('constructor of KNN graph Classifier')

    def train(self,training_data):
        """ train the classification model with training dataset

            Attributes:
            trainig_data: dataset used for training the classifier
        """

        print('train method of graph KNN Classifier')


    def test(self, testing_data):
        """ test the classification model with testing dataset
            Attributes:
            testing_data: dataset used for testing the classifier
        """

        print('train method of graph KNN Classifier')


#-------------- Class Experiment --------------#

class Experiment:
    """ Represents methods for run selected classifier with selected DataSet.

        Methods:
        __init__: constructor of the class
        runCrossVal: evaluate classifier on a given data sample
        score: generates classifier's accuracy result as a numeric,
        range from 0 to 100.
        __confusionMatrix: construct a confusion matrix for measuring the
        performance of the classification model

    """
    def __init__(self, dataset='', classifier=''):
        """ Create an instance of the Experiment class.

        Attributes:
        dataset: takes in a 2 dimensional dataset for model training and testing
        classifier: choose a classfication method presented in
        the ClassiferAlgorithm class

        """

        self.dataset = dataset
        self.classifier = classifier

        print('constructor of Experiment')


    def runCrossVal(self, k):

        """
        run a k-fold cross validation of the model chosen

        Attributes:
        k: int, number of groups that a given data sample is to be split into
        """

        print('runCrossVal method')


    def score(self):
        """
        generates the given classifier's accuracy result as a numeric score,
        range from 0 to 100.

        """

        print( 'score method')

    def __confusionMatrix(self):

        """
        construct a confusion matrix for measuring the
        performance of the classification model.

        """

        print('__confusionMatrix method')
