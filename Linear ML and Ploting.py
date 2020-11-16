import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn.utils import shuffle


data = pd.read_csv("student-mat.csv", sep=";")  #setup data, grab relevant columns
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"  #this is our label, what we want to predict/get, can have multiple if you want

x = np.array(data.drop([predict], 1))       #returns new data frame without G3, so creates training data
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) #redefined up here so can still use it when done finding best model

"""
best = 85
for _ in range(300):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)    #taking all our attributes and labels(prediction) and split into 4 arrays, 0.1 means that we are only using 10% of all data to train
#x and y train are sections of above x and y arrays, we use xandy test to test our model
#if we train model on every piece of data, computer already sees info so knows the answer already, so cant use it in future or on other samples

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train) #creates line of best fit for x and y train, then stores it in linear
    acc = linear.score(x_test, y_test)  #returns value that is accuracy of model
    print("here it comes")
    print(acc) #we got 81% for first run, everytime you run it it changes lol
    if acc > best:
        best = acc #updates our best model
        with open("studentmodel.pickle", "wb") as f:#Save as pickle file in directory
            pickle.dump(linear, f)

"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)         #loads the pickle file into linear model so uses that pickle model instead of relearning

#now to save a model until we get a certain score


#Now to actually use and test the model on other data


#shows the linear coefficients and intercept
print("Co: ", linear.coef_)
print("intercerpt: " , linear.intercept_)

predictions = linear.predict(x_test)    #

#print out all predictions and show input data for all predictions
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x]) #shows us the prediction, then the variables being tested, then the actual value to compare to

#Now how to save, plot data, visualize what we are doing
#You save models 1 because they take a while to train 2 because you find one with high accuracy


p = "G1" #can change this and see correlation between different points, maybe see if P is affecting final grade
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
