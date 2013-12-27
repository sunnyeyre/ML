
import csv as csv
import numpy as np

class enum(object):
   def __init__(self, names):
      for number, name in enumerate(names.split(',')):
         setattr(self, name, number)

def load_data(filename, ext):
   csv_file_object = csv.reader(open("%s" % filename, 'rb')) #Load file
   header = csv_file_object.next() #save the first line in header
   data = []
   for row in csv_file_object:
      if ext == 'train':
         data.append(row[1:])
      else:
         data.append(row[0:])
   data = np.array(data)

   return data



