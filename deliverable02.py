"""
/*
* deliverable02_final_v3.py
*
* ANLY 555 Fall 2020
* Project Toolboox Dilverable
*
* Due on: 10/11/2020
* Author: Leyao Li    Lin Meng
*
*
* In accordance with the class policies and Georgetown's
* Honor Code, I certify that, with the exception of the
* class resources and those items noted below, I have neither
* given nor received any assistance on this project other than
* the TAs, professor, textbook and teammates.
*
* References not otherwise commented within the program source code.
* Note that you should not mention any help from the TAs, the professor,
* or any code taken from the class textbooks.
*/
"""



# import necessary packages

import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



#-------------- Class DataSet --------------#
class DataSet:
    """ Represent basic information for a two-dimensional dataset in tabular form
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file
        __load: a more generic method for reading in the dataset from a file
                will ask for user input of dataset type and file name
        clean: handles N/A values
        explore: show summary statistics
                 plot 2 visualizations of the dataset
    """


    def __init__(self, filename=None ):
        """ Create an instance of a DataSet

            Attributes:
            filename: name of the file

        """

        self.filename = filename
#         self.__load(filename)
#         self.__readFromCSV(filename)

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
        """ handle N/A values in the dataset or remove noise from the dataset

            handling method will be specific to each type of data
            the dataset contains

        """
        print ('main clean method')


    def explore(self):
        """ show summary statistics
            and generate 2 visualizations of the dataset
        """

        print('main explore method')

#-------------- Class QuantDataSet  --------------#
class QuantDataSet(DataSet):
    """ Represent a two-dimensional dataset with quantitative values
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file(inherit from supperclass)
        __load: a more generic method for reading in the dataset from a file
                will ask for user input of dataset type and file name (inherit from supperclass)
        clean: handles N/A values with mean
        explore: 1. plot a box plot which can show people the summary of each variable like mean, min, max and etc.
                 2. plot a line plot which can show people the data is increase or decrease.

    """
    def __init__(self,filename=None):
        """ Create an instance of a QuantDataSet
            Attributes:
            filename: path to where the file is located. The DataSet instance
            will be populated using the given file.
            If the filename is not given, an empty dataset will be created.
        """
        super().__init__(filename)
        
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
        """ handle N/A values in the dataset
            by filling in with mean
        """
        data=np.array(self.my_data)
        col=data[1:,0]
        row=data[0,:]
        self.value=np.delete(data,0,0)
        self.value=np.delete(self.value,0,1)
        self.value=self.value.astype(float)
        col_mean=np.nanmean(self.value,axis=0)
        inds=np.where(np.isnan(self.value))
        self.value[inds]=np.take(col_mean,inds[1])
        cc=np.concatenate((col[:,None],self.value),axis=1)
        self.my_data=np.concatenate((row[None,:],cc),axis=0)
        return self.my_data

    def explore(self,col_row,x):
        """
        explore: 1. plot a box plot which can show people the summary of each variable like mean, min, max and etc.
                 2. plot a line plot which can show people the data is increase or decrease.
        Attributes:
            x: could be int number. Let people to choose which line they want to load.
            col_row: enter 'col' or 'row' to choose the way people want to analyses data by columns or rows.

        """
        #convert value to numpy array and keep numeric data only
        data=np.array(self.my_data)
        self.value=np.delete(data,0,0)
        self.value=np.delete(self.value,0,1)
        self.value=self.value.astype(float)
        # set a a if conition to analyses data by columns
        if col_row=='col':
            #set a if condition to make sure attributes is numeric
            if type(x)==int:
                x=[x]
                x1=[]
                for i in x:
                    i=i-1
                    x1.append(i)
                p=self.value[:,x1]
                name=list(data[0,x])
                p1=list(self.value[x1,:])
                p2=np.array(range(1,p.shape[0]+1))
                p3=np.concatenate((p2[:,None],p),axis=1)
                #Show the box plot
                plt.title('Box Plot')
                plt.boxplot(p1)
                plt.xticks(x,name)
                plt.show()
                #Show the line plot
                plt.title('Line Plot')
                plt.plot(p3[:,0],p3[:,1])
                plt.legend(name)
                plt.show()
            #if attribute is not numeric, print X should be int.
            else:
                print("X should be int")
        # set a a if conition to analyses data by rows
        elif col_row=='row':
            if type(x)==int:
                x=[x]
                x1=[]
                for i in x:
                    i=i-1
                    x1.append(i)
                p=self.value[x1,:]
                name=list(self.my_data[x1,0])
                p1=list(self.value[x1,:])
                p2=np.array(range(1,p.shape[1]+1))
                p3=np.concatenate((p2[None,:],p1),axis=0)
                plt.title('Box Plot')
                plt.boxplot(p1)
                plt.xticks(x,name)
                plt.show()
                plt.title('Line Plot')
                plt.plot(p3[0,:],p3[1,:])
                plt.legend(name)
                plt.show()
            ##if attribute is not numeric, print X should be int.
            else:
                print("X should be int")
        else:
            print("col_row should be only 'col' or 'row ")

#-------------- Class QualitDataSet  --------------#
class QualitDataSet(DataSet):
    """ Represent a two-dimensional dataset with qualitative values
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file(inherit from supper class)
        __load: a more generic method for reading in the dataset from a file
                will ask for user input of dataset type and file name(inherit from superclass)
        clean: handles N/A values with mode
        explore: 1. plot a pie plot which can show people the percentage of each category.
                 2. plot a bar plot which can sow the specific number of each category

    """
    def __init__(self,filename=None):
        """ Create an instance of a QuantDataSet
            Attributes:
            filename: path to where the file is located. The DataSet instance
            will be populated using the given file.
            If the filename is not given, an empty dataset will be created.

        """
        super().__init__(filename)

    def clean(self):
        """ handle N/A values in the dataset
            by filling in with mode
        """
        self.value=np.array(self.my_data)
        for i in range(self.value.shape[1]):
            a=list(self.value[:,i])
            count=0
            num=a[0]
            for j in a:
                f=a.count(j)
                if(f>count):
                    count=f
                    num=j
                    for k in range(len(a)):
                        if(a[k]==''):
                            a[k]=num
                            self.value[:,i]=a
        self.my_data=self.value
        return self.my_data

    def explore(self,x):
        """
        explore: 1. plot a pie plot which can show people the percentage of each category.
                 2. plot a bar plot which can sow the specific number of each category
        Attributes:
            x: could be int number or a int list. Let people to choose which variable or variables they want to explore.

        """
        self.my_data=np.array(self.my_data)
        #set a if condition to know people imput int or list
        if type(x)==list:
            #count the frequency of each category in the variable
            for k in x:
                p=self.my_data[:,k]
                name=[]
                number=[]
                count_data=[]
                for j in p:
                    f=p.count(j)
                    if j not in name:
                        name.append(j)
                        number.append(f)
                for n in range(len(name)):
                    b=[name[n],number[n]]
                    count_data.append(b)
            #show the pie plot
            plt.title("Pie Plot")
            plt.pie(number,labels=name)
            plt.show()
            #show the bar plot
            plt.title("Bar Plot")
            plt.bar(name,number)
            plt.show()
        #set a if condition to know people imput int or list
        elif type(x)==int:
            p=list(self.my_data[:,x])
            name=[]
            number=[]
            count_data=[]
            #count the frequency of each category in the variable
            for j in p:
                f=p.count(j)
                if j not in name:
                    name.append(j)
                    number.append(f)
                for n in range(len(name)):
                    b=[name[n],number[n]]
                    count_data.append(b)
            #show the pie plot
            plt.title("Pie Plot")
            plt.pie(number,labels=name)
            plt.show()
            #show the bar plot
            plt.title("Bar Plot")
            plt.bar(name,number)
            plt.show()
        else:
            print("x should be int")
#-------------- Class TextDataSet  --------------#
class TextDataSet(DataSet):
    """ Represent a two-dimensional dataset with quantitative values
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file(inherit from supperclass)
        __load: a more generic method for reading in the dataset from a file
                will ask for user input of dataset type and file name (inherit from supperclass)
        clean: remove stopwords which is meaningless
        explore: 1. plot a bar plot which can show people the occurrence of each world
                 2. plot words cloud which can show people the world which have occurrence a lot of time.
    """

    def __init__(self,filename=None):
        super().__init__(filename)

    def clean(self):
        """ delete the stopwords which is meanless.
        """
        self.text=self.my_data
        self.text.remove(self.text[0])
        stopword=stopwords.words('english')
        cleaned=[self.my_data[0]]
        for i in range(len(self.text)):
            afc=[]
            ctext=self.text[i]
            tokens=word_tokenize(str(ctext))
            for j in tokens:
                if j not in stopword:
                    afc.append(j)
            cleaned.append(afc)
        self.my_data=cleaned
        return self.my_data


    def explore(self,x):
        """
        explore: 1. plot a worldcloud which can show people the word which has larage occurrence.
                 2. plot a bar plot which can sow the specific number of each word.
        Attributes:
            x: could be int number or a int list. Let people to choose which variable or variables they want to explore.

        """
        #set a if condition to know people imput int or list
        if type(x)==list:
            p=[]
            for i in x:
                c=self.my_data[i]
                p.append(c)
            #count the frequency of each word in the text
            for i in range(len(p)):
                name=[]
                number=[]
                ctext=p[i]
                for j in ctext:
                    f=ctext.count(j)
                    if j not in name:
                        name.append(j)
                        number.append(f)
                count_data=[]
                for n in range(len(name)):
                    a=[name[n],number[n]]
                    count_data.append(a)
                #show the bar plot of each text
                plt.title("Bar Plot")
                plt.bar(name,number)
                plt.show()
                #show the world cloud of each text
                plt.title('world cloud')
                cloud=WordCloud().generate(str(ctext))
                plt.imshow(cloud)
                plt.show()
        #set a if condition to know people imput int or list
        elif type(x)==int:
            p=self.my_data[x]
            name=[]
            number=[]
            #count the frequency of each word in the text
            for j in p:
                f=p.count(j)
                if j not in name:
                    name.append(j)
                    number.append(f)
            count_data=[]
            for n in range(len(name)):
                a=[name[n],number[n]]
                count_data.append(a)
            plt.title("Bar Plot")
            plt.bar(name,number)
            plt.show()
            plt.title('world cloud')
            cloud=WordCloud().generate(str(p))
            plt.imshow(cloud)
            plt.show()

        else:
            print("x should be int or list")

#-------------- Class TimeSeriesDataSet  --------------#

class TimeSeriesDataSet(DataSet):
    """ Represent a two-dimensional dataset indexed in time order
        Inherit from the superclass DataSet
        Methods:
        __init__: constructor, create an instance of the class
        __readFromCSV: read in the dataset from a csv file
        __load: a more generic method for reading in the dataset from a file
        clean: handles N/A values
        explore: autocorrelation,
    """

    def __init__(self, filename=None ):
        """ Create an instance of a TimeSeriesDataSet



            Attributes:
            filename: path to where the file is located. The DataSet instance
            will be populated using the given file.
            If the filename is not given, an empty dataset will be created.

        """
        super().__init__(self)
        self.__load(filename)


    def __readFromCSV(self, filename, header=False):
        """ Read in the dataset that is in a CSV file
            Assume all datatypes are numeric
            Parameter:
            header: bool, asks if the file has hearders
        """
        try:
            with open(os.getcwd()+'/'+filename,'r') as file:
                self.my_reader = csv.reader(file, delimiter = ',')
                self.my_data = list(self.my_reader)
                self.my_data_array = np.array(self.my_data).astype(np.float)
        except:
            print("This file is not a CSV file.\n")




    def __load(self, filename):
        """ Load in the file


        """
        if not filename:
                filename = input('Enter your filename to be loaded: ')
        print(os.getcwd()+'/'+filename)
        if os.path.exists(os.getcwd()+'/'+filename):
            self.__readFromCSV(filename)
        else:
            print('The file does not exist in the current working directory.')


    def medfilt (self, data_array, k):
        """ Define a median filter using numpy for filtering noise in
            time series data
            data array will be padded with zeros

        """
        assert (k - 1) % 2 == 0 # k must be an odd number
        k2 = (k - 1) // 2  # median index within each window
        filtered_array = np.zeros(len(data_array))
        for i in range(k2, len(data_array)-k2):
            filtered_array[i] = np.median(data_array[i-k2:i+k2])
        return filtered_array



    def clean(self,k=3):
        """ Apply a median filter of size n to a 1D data_array.
            Final result will be padded with zeros.
            k = 3 is the default median filter size


        """

        for i in range(len(self.my_data_array)):
            self.my_data_array[i] = self.medfilt(self.my_data_array[i],k)






    def explore(self):
        """ show summary statistics of the dataset through plots
            1. scatter plot
            2. autocorrelation
        """
#       visualization 1: scatter plot
#       show average singal intensity of all samples at each time point
        data_array_avg = self.my_data_array.mean(axis=0)
        plot1 = plt.figure(1)
        plt.title('avegrage signal intensity of all samples')
        plt.xlabel('time')
        plt.ylabel('signal')

        plt.plot(data_array_avg)


#       visualization 2: autocorrelation plot
#       show correlation coefficient scores for the time series data
        plot2 = plt.figure(2)
        plt.acorr(data_array_avg)
        plt.title('autocorrelation plot')
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.show()
