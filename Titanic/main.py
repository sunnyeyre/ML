from Loader import *
from KNN import *
from DecisionTree import *
import numpy as np
import csv as csv
from NaiveBayes import *

test_columns = enum("PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked")

def predict(myRow):
   if myRow[3] == 'female':
      return 1
   else:
      return 0

def bin_data(data): #bin age into 5 groups and fare into 4
   for row in data:
      for i,v in enumerate(att_values[tc.Age]):
         if (row[tc.Age] == ''):
            break
         if (float(row[tc.Age]) <= float(v)):
            row[tc.Age] = v
            break
      for i,v in enumerate(att_values[tc.Fare]):
         if (row[tc.Fare] == ''):
            if int(row[tc.Pclass]) == 3:        # in which case put in std fare
               row[tc.Fare] = '7.25'
            elif int(row[tc.Pclass]) == 2:
               row[tc.Fare] = '15.0'
            elif int(row[tc.Pclass]) == 1:
               row[tc.Fare] = '30.0'
         if (float(row[tc.Fare]) <= float(v)):
            row[tc.Fare] = v
            break

def output_test_file(input_filename, output_filename):
   #class, gender, and ticket fare
#   KNN_classifier = KNN(5, [test_columns.Pclass,test_columns.Sex,test_columns.Fare])
   train_data = load_data('train.csv', 'train')

   bin_data(train_data)
#   attributes = [ x for x,y in enumerate(att_values) if (y != 'skip' and x != 0)]
#   DecisionTreeClassifier = DecisionTree(train_data, attributes,'')
   NBClassifier = \
   NaiveBayes([test_columns.PassengerId,test_columns.Sex,test_columns.Fare,test_columns.Pclass,test_columns.Age])

   test_data = load_data(input_filename, 'test');
   output_file_object = csv.writer(open("%s" % output_filename, 'wb'))
   output_file_object.writerow(["Survived", "PassengerID"])

#   for row in test_data:
#      if row[test_columns.Sex] == 'female':
#         row[test_columns.Sex] = 0.0
#      else:
#         row[test_columns.Sex] = 1.0

   bin_data(test_data)
   for row in test_data:
      if NBClassifier.predict(row) == 1:
         output_file_object.writerow(["1", row[0]])
      else:
         output_file_object.writerow(["0", row[0]])

output_test_file('test.csv', 'NB_testpython.csv')
