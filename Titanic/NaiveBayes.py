from Loader import *
import numpy as np
import csv as csv
import math as math

'''''
   Naive Bayes
   probability estimate: (nc + m*p)/(n + m), nc = number of samples labeled with c, n = total number
   of samples, p = 1/k, k = number of possible values, m = constant = n/k
   > training is to learn the probabilities P(s)
   > classification is to calculate P(v)PI(P(v|s)P(s)) for all possible values s
   > n bins (n for each possible value of ta) x m attributes => use map reduce
'''''

tc = enum("Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked")

class NaiveBayes(object):
   def __init__(self, attributes):
      self.data = load_data('train.csv', 'train')
      self.probability_table = [x for x in xrange(len(self.data[0,0::]))] # array of hashes keyed by attributes
      self.conditional_plus = [x for x in xrange(len(self.data[0,0::]))]
      self.conditional_minus = [x for x in xrange(len(self.data[0,0::]))]
      for i in xrange(len(self.data[0,0::])):
         self.probability_table[i] = {}
         self.conditional_plus[i] = {}
         self.conditional_minus[i] = {}
      self.attributes = attributes #array of enums of attributes we are considering

      self.train()

   def train(self):
   # this gives P(v) and P(s)
      for row in self.data:
         for i in self.attributes:
            if row[i] not in self.probability_table[i]:
               self.probability_table[i][row[i]] = 1
            else:
               self.probability_table[i][row[i]] += 1
            if row[tc.Survived] == '1': # this gives P(v|s)
               if row[i] not in self.conditional_plus[i]:
                  self.conditional_plus[i][row[i]] = 1
               else:
                  self.conditional_plus[i][row[i]] += 1
            else:
               if row[i] not in self.conditional_minus[i]:
                  self.conditional_minus[i][row[i]] = 1
               else:
                  self.conditional_minus[i][row[i]] += 1
      total = np.size(self.data[0::,0].astype(np.float))
      for i in self.attributes:
         for k,v in self.probability_table[i].iteritems():
            self.probability_table[i][k] = float(v) / total
         n_plus = len(self.conditional_plus[i].keys())
         n_minus = len(self.conditional_minus[i].keys())
         for k,v in self.conditional_plus[i].iteritems():
            self.conditional_plus[i][k] = (float(v) + 1.0)/ (total + float(n_plus))
         for k,v in self.conditional_minus[i].iteritems():
            self.conditional_minus[i][k] = (float(v) + 1.0)/ (total + float(n_minus))

   def predict(self,row):
      plus = float(self.probability_table[tc.Survived]['1'])
      minus = float(self.probability_table[tc.Survived]['0'])
      total = np.size(self.data[0::,0].astype(np.float))
      for i in self.attributes:
         if i != tc.Survived:
            if row[i] not in self.conditional_plus[i]:
               n_plus = len(self.conditional_plus[i].keys()) + 1
               self.conditional_plus[i][row[i]] = 1.0 / (total + float(n_plus))
            if row[i] not in self.conditional_minus[i]:
               n_minus = len(self.conditional_minus[i].keys()) + 1
               self.conditional_minus[i][row[i]] = 1.0 / (total + float(n_minus))
            if row[i] in self.probability_table[i]:
               plus *= float(self.conditional_plus[i][row[i]]) * self.probability_table[i][row[i]]
               minus *= float(self.conditional_minus[i][row[i]]) * self.probability_table[i][row[i]]

      if plus > minus:
         return 1
      else:
         return 0


