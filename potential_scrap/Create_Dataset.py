import os
import string

import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import OneHotEncoder

# Random seed for reproducibility
np.random.seed(42)

categories = [[l] for l in list(string.ascii_uppercase)]
categories.insert(0, ['0']) #Creates the classes required for the sign language data
# define one hot encoding
encoder = OneHotEncoder(sparse_output=False)
# transform data
onehot = encoder.fit_transform(categories) #Fits each letter A,B,C -> a one hot

d = {} #Create dictionary
d['0'] = onehot[0]
i=1
for k in list(string.ascii_uppercase): #Creates key:pair dictionary of each Category and its oneHot
    d[k] = onehot[i]
    i+=1


def listdir_nohidden(path): #Removes any path that has a . at the front (directory cleanup)
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def create_dataset(img_folder): #Creates a dataset
    '''Function that takes a path for a folder of folders of images (give it path to
    data or dataCT etc)'''
    im = [] #List of image values
    class_name = []#Corresponding one hot class


    for sub_folder in listdir_nohidden(img_folder): #Gets each folder within data folder
        for file in listdir_nohidden(os.path.join(img_folder,sub_folder)): #Gets 'file' i.e. individual image in folder

            image = cv2.imread(os.path.join(img_folder,sub_folder,file)) #read image
            # image = np.array(image) #turn image into an array
            # image = image.astype('float32') #sets datatype of array
            # image /= 255 #normalise
            im.append(image)
            class_name.append(d[sub_folder])

    x_train, x_test, y_train, y_test = train_test_split(im, class_name, test_size = 0.33)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    
    return x_train, x_test, y_train, y_test,d
