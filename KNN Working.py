# K - nearest numbers

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data") #pandas reads commas as separators for columns, and first line as column names
print(data.head())

#with non numerical data, we must convert into numerical data, we use preprocessing

le = preprocessing.LabelEncoder() #takes labels and encodes them into appropriate integer values

buying = le.fit_transform(list(data["buying"]))     #gets all buying column and gets into appropriate integer values
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety)) #zip creates tuple objects with lists that we give it
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) #redefined up here so can still use it when done finding best model

model = KNeighborsClassifier(n_neighbors=9) #how many closest neighbors do you want?
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    #model.kneighbors([x_test[x]], 9, True) #extra brackets so xtest comes in as 2d value, this shows the distance so you can plot the data if you are stupid boy
    #print("N: ", n)

