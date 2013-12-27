from Loader import *
import numpy as np
import math as math
import csv as csv

''''' 
   KNN = Sigma(w(i)*training_predict(i))/Sigma(w(i)) for top k nieghbors and take average
   w(i) = 1/(d(i) - d(test,i))^2, d = float/discrete values
   take care of special case, delta(d) = 0
   also put in w(i) = e^(-2d^2)/sigma^2 = Gaussian
'''''

tc = enum("Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked")

class KNN(object):
   def __init__(self, k, columns):
      self.columns = columns #attributes of the dataset used
      self.k = k
      self.data = load_data('train.csv', 'train')
      self.max_columns = np.zeros(np.size(self.data[0,0::]),float)
      self.min_columns = np.ones(np.size(self.data[0,0::]),float) * 10000.0

      for row in self.data:
         if (row[tc.Sex] == 'female'): #change female to 0 and male to 1
            row[tc.Sex] = 0.0
         else:
            row[tc.Sex] = 1.0
         if (row[tc.Fare] == ''): #class fare, can be empty string
            if int(row[self.columns[tc.Pclass]]) == 3:        # in which case put in std fare
               row[tc.Fare] = 7.25
            elif int(row[self.columns[tc.Pclass]]) == 2:
               row[tc.Fare] = 15.0
            elif int(row[self.columns[tc.Pclass]]) == 1:
               row[tc.Fare] = 30.0
         for j in self.columns: #find min and max for norm constant
            if (row[j] != ''):
               if (float(row[j]) > self.max_columns[j]):
                  self.max_columns[j] = float(row[j])
               elif (float(row[j]) < self.min_columns[j]):
                  self.min_columns[j] = float(row[j])
      
      self.train_size = np.size(self.data[0::,0].astype(np.float))

   def predict(self,Row):
      heapQ = []
      for i in range(self.train_size):
         delta = 0.0
         norm = float(len(self.columns))
         for j in self.columns:
            try:
               value = (float(Row[j]) - float(self.data[i,j]))*(float(Row[j]) - float(self.data[i,j]))
            except:
               if (Row[j] == '') and (j == tc.Fare): #class fare, can be empty string
                  if int(float(Row[tc.Pclass])) == 3:        # in which case put in std fare
                     Row[j] = 7.25
                     value= (float(Row[j]) - float(self.data[i,j]))*(float(Row[j]) - float(self.data[i,j]))
                  elif int(float(Row[tc.Pclass])) == 2:
                     Row[j] = 15.0
                     value= (float(Row[j]) - float(self.data[i,j]))*(float(Row[j]) - float(self.data[i,j]))
                  elif int(float(Row[tc.Pclass])) == 1:
                     Row[j] = 30.0
                     value= (float(Row[j]) - float(self.data[i,j]))*(float(Row[j]) - float(self.data[i,j]))
                  else:
                     print "there is a non Pclass! %s" %(Row[tc.Pclass])
                     break
               elif (j == tc.Age) or (j == mtc.SibSp):
                  norm -= 1.0
                  continue
               else: 
                  print "other exceptions! Row[j] %s self.data[i,j] %s" %(Row[j],self.data[i,j])
                  break
            if (float(self.max_columns[j]) - float(Row[j]))*(float(self.max_columns[j])-float(Row[j])) > \
               (float(self.min_columns[j]) - float(Row[j]))*(float(self.min_columns[j])-float(Row[j])):
               value = value / ((float(self.max_columns[j]) -
               float(Row[j]))*(float(self.max_columns[j])-float(Row[j])))
               delta += value
            else:
               value = value /((float(self.min_columns[j]) -
               float(Row[j]))*(float(self.min_columns[j])-float(Row[j])))
               delta += value
            #divide by normalization constant 
         delta /= norm
         heapQ.append((delta, float(self.data[i,tc.Survived])))
                  
      if sum(pair[1] for pair in sorted(heapQ)[:self.k])/float(self.k) > 0.5:
         return 1.0
      else:
         return 0.0

   def predict_Gaussian(self,Row):
      heapQ = []
      for i in range(self.train_size):
         delta = 0.0
         norm = float(len(self.columns))
         for j in self.columns:
            try:
               value = (float(Row[j]) - float(self.data[i,j]))*(float(Row[j]) - float(self.data[i,j]))
            except:
               if (Row[j] == '') and (j == tc.Fare): #class fare, can be empty string
                  if int(float(Row[tc.Pclass])) == 3:        # in which case put in std fare
                     Row[j] = 7.25
                     value= (float(Row[j]) - float(self.data[i,j]))*(float(Row[j]) - float(self.data[i,j]))
                  elif int(float(Row[tc.Pclass])) == 2:
                     Row[j] = 15.0
                     value= (float(Row[j]) - float(self.data[i,j]))*(float(Row[j]) - float(self.data[i,j]))
                  elif int(float(Row[tc.Pclass])) == 1:
                     Row[j] = 30.0
                     value= (float(Row[j]) - float(self.data[i,j]))*(float(Row[j]) - float(self.data[i,j]))
                  else:
                     print "there is a non Pclass! %s" %(Row[tc.Pclass])
                     break
               elif (j == tc.Age) or (j == mtc.SibSp):
                  norm -= 1.0
                  continue
               else: 
                  print "other exceptions! Row[j] %s self.data[i,j] %s" %(Row[j],self.data[i,j])
                  break
            if (float(self.max_columns[j]) - float(Row[j]))*(float(self.max_columns[j])-float(Row[j])) > \
               (float(self.min_columns[j]) - float(Row[j]))*(float(self.min_columns[j])-float(Row[j])):
               value = value / ((float(self.max_columns[j]) -
               float(Row[j]))*(float(self.max_columns[j])-float(Row[j])))
               value = math.exp(-value)/ math.exp(1)
               delta += value
            else:
               value = value /((float(self.min_columns[j]) -
               float(Row[j]))*(float(self.min_columns[j])-float(Row[j])))
               value = math.exp(-value) / math.exp(1)
               delta += value
            #divide by normalization constant 
         delta /= norm
         heapQ.append((delta, float(self.data[i,tc.Survived])))
                  
      if sum(pair[1] for pair in sorted(heapQ)[:self.k])/float(self.k) > 0.5:
         return 1.0
      else:
         return 0.0


#print train_data[0::,tc.Pclass]
#number_passengers = np.size(train_data[0::,0].astype(np.float))
#number_survived = np.sum(train_data[0::,0].astype(np.float))

