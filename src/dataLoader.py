import pickle as cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('C:/Users/dennis/Documents/HandWrittenAI/data/mnist.pkl.gz', 'rb')
    u=cPickle._Unpickler(f)    
    u.encoding='latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)


def prepareForTraining():
    training_data,val_data,test_data=load_data()
    final_training_set=[]
    final_test_set=[]
    training_set=list(zip(training_data[0],training_data[1]))
    test_data=list(zip(test_data[0],test_data[1]))
    for entry in training_set:
        set_train=[]
        vectorizedInput=np.array(entry[0]).reshape(784,1)
        set_train.append(vectorizedInput)
        set_train.append(outputVector(entry[1]))
        final_training_set.append(set_train)
    for entry in test_data:
        set_train=[]
        vectorizedInput=np.array(entry[0]).reshape(784,1)
        set_train.append(vectorizedInput)
        set_train.append([entry[1]])
        final_test_set.append(set_train)
    
    return [final_training_set,final_test_set]

def outputVector(number):
    newOutputVector=np.zeros([10,1])
    newOutputVector[number]=1.0
    return newOutputVector

prepareForTraining()