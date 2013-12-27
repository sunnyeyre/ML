from Loader import *
import numpy as np
import csv as csv
import math as math

'''''
   ID3(Examples, Target_Attribute, Attributes)
   create a root node from the tree. If all Examples are +, return the single node with +, - ditto.
   If Attributes is empty, return Root with label = most common value of Target_attribute in
   Examples. Otherwise 
   - pick A as attribute with highest information gain (entropy) or Info gain ratio (split info)
      info_gain = Entropy(s) - Sum(s(v)/s * Entropy(sv)), 
      Entropy = -(+/total)log(+/total) - (-/total)log(-/total)
   - for each possible value of A 
      1) Add a new tree branch below Root with A=vi
      2) Let Examples(vi) be subset of Examples that have vi for A
      3) if Examples(vi) is empty - add a leaf node with label = most common value of
      Target_attribute in examples, else add subtree ID3(Examples(vi), Target_attribute,
      Attributes-{A})
   - stopping criterian vs rule postpruning : right now stopping when gain ratio stops increasing
'''''

tc = enum("Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked")
att_values = ( ('0','1'), ('1','2','3'), 'skip', ('male', 'female'), ('18','29','39','49','100'), ('1','0'),
('0','1','2'), 'skip',('10','20','30','1000'), 'skip', ('C','Q','S'))

class DecisionTree(object):
   def __init__(self, examples, attributes, label): #ta = target_attribute = survived in this case
      self.ta = tc.Survived
      self.attributes = set(attributes) #attributes of the dataset used
      self.examples = examples # examples = parent.data[ (data[0::,tc.A == "vi"]), 0]
      self.children = {} # Decision Trees keyed by their attribute values
      self.splitting_attribute = ''
      self.label = label

      self.build_tree()
   
   def count_examples(self,examples):
      positive = np.sum(examples[ (examples[0::,self.ta] == '1'), 0].astype(np.float))
      negative = np.size(examples[0::, self.ta]) - positive
      return (positive, negative)

   def entropy(self,examples):
      (positive, negative) = self.count_examples(examples)
      total = positive + negative
      if (positive == 0) or (negative == 0):
         return 0.0
      return -(positive/total)*math.log(positive/total,2) - (negative/total)*math.log(negative/total,2)

   def get_gain_n_gain_ratio(self, attribute_enum, attribute_values, examples): #need a list of attribute_values
      S = []
      E = []
      for i,v in enumerate(attribute_values):
         S.append(np.sum(examples[ (examples[0::, int(attribute_enum)] == v), 0].astype(np.float)))
         E.append( self.entropy(examples[ (examples[0::, int(attribute_enum)] == v)]))
      total = sum(S)
      if (total == 0.0):
         return (0.0, 0.0)
      split_info = - sum(map(lambda x:(x/total)*math.log(x/total,2), (x for x in S if x > 0.0)))
      gain = self.entropy(examples) - sum( int(x)*y for x,y in zip(S,E)) / total
      if (split_info <= 0.0):
         return (gain, gain)
      else:
         return (gain, gain/split_info)
   
   def build_tree(self): # 0 for gain and 1 for gain ratio
      max_att = (10000.0, 0.0)
      if self.label != '': #no need to process a leaf
         return
      if not len(self.attributes):
         (positive, negative) = self.count_examples(self.examples)
         if positive > negative:
            self.label = '+'
         else:
            self.label = '-'
         return
      for att in self.attributes: #find best attribute to branch
         (gain, gain_ratio) = self.get_gain_n_gain_ratio(att, att_values[att], self.examples)
         if gain > max_att[1]:
            max_att = (att, gain)
      if max_att[0] == 10000.0: # if gain_ratio does not increase, pick the first one
         (positive, negative) = self.count_examples(self.examples)
         if positive >= negative:
            self.label = '+'
         else:
            self.label = '-'
         return
      self.splitting_attribute = max_att[0]
      for att_value in att_values[max_att[0]]:
         if not len(self.examples[ (self.examples[0::, max_att[0]] == att_value)]):
            #if there are no more examples, assign a leaf with majority label of parent
            (positive, negative) = self.count_examples(self.examples)
            if (positive >= negative):
               self.children[att_value] = DecisionTree(None, self.attributes - set([max_att[0]]),
               '+')
            else:
               self.children[att_value] = DecisionTree(None, self.attributes - set([max_att[0]]),
               '-')
         else:
            self.children[att_value] = DecisionTree(self.examples[self.examples[0::, max_att[0]] ==
            att_value], self.attributes - set([max_att[0]]), '')
      
   def predict(self, row): #walk down Decision Tree and get the label from the leaf at end of walk
      if (self.label == ''):
         print 'going down one more layer from %d %s' %(self.splitting_attribute,
         row[self.splitting_attribute])
         try: #if getting a test value not in the data, get most prevalent label from the branch
            return self.children[row[self.splitting_attribute]].predict(row)
         except:
            if not self.children[row[self.splitting_attribute]]:
               (positive, negative) = self.count_examples[self.examples]
               if positive > negative:
                  return 1
               else:
                  return 0
      else:
         if (self.label == '+'):
            print 'label is +'
            return 1
         else:
            print 'label is -'
            return 0
   def walktree(self):
      if self.label != '':
         print self.label
      else:
         print self.splitting_attribute
      for k in self.children.keys():
         self.children[k].walktree()

