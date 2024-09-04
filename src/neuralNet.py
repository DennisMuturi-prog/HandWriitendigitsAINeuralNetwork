from dataLoader import prepareForTraining
import numpy as np
import random
from mnist_loader import load_data_wrapper
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z)) 
def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

# print(sigmoid(np.array([[1],[2],[3],[4]])))
class NeuralNetwork:
    def __init__(self,structure):
        self.structure=structure
        self.neuralNet=structure
        self.w1=np.random.randn(structure[1],structure[0])
        self.w2=np.random.randn(structure[2],structure[1])
        self.w3=np.random.randn(structure[3],structure[2])
        self.b1=np.random.randn(structure[1],1)
        self.b2=np.random.randn(structure[2],1)
        self.b3=np.random.randn(structure[3],1)
        self.a0=[]
        self.a1=[]
        self.a2=[]
        self.a3=[]
        self.z1=[]
        self.z2=[]
        self.z3=[]
        self.a4=[]
        self.sumGradient_w1=np.zeros([structure[1],structure[0]])
        self.sumGradient_b1=np.zeros([structure[1],1])
        self.sumGradient_w2=np.zeros([structure[2],structure[1]])
        self.sumGradient_b2=np.zeros([structure[2],1])
        self.sumGradient_w3=np.zeros([structure[3],structure[2]])
        self.sumGradient_b3=np.zeros([structure[3],1])



    
    def feedForward(self,a0):
        self.a0=a0
        self.z1=(self.w1@self.a0)+self.b1
        self.a1=sigmoid(self.z1)
        self.z2=(self.w2@self.a1)+self.b2
        self.a2=sigmoid(self.z2)
        self.z3=(self.w3@self.a2)+self.b3
        self.a3=sigmoid(self.z3)
        output=np.array(self.a3)
        return output
    
    def backwardPropagation(self,desiredOutput):
        gradient_a3=2*(self.a3-desiredOutput)
        gradient_b3=gradient_a3*sigmoid_prime(self.z3)
        gradient_w3=gradient_b3@self.a2.T
        gradient_b2=(((gradient_b3.T@self.w3).T)*sigmoid_prime(self.z2))
        gradient_w2=gradient_b2@self.a1.T
        gradient_b1=(((gradient_b2.T@self.w2).T)*sigmoid_prime(self.z1))
        gradient_w1=gradient_b1@self.a0.T
        self.sumGradient_b3+=gradient_b3
        self.sumGradient_w3+=gradient_w3
        self.sumGradient_b2+=gradient_b2
        self.sumGradient_w2+=gradient_w2
        self.sumGradient_b1+=gradient_b1
        self.sumGradient_w1+=gradient_w1

    def tuneNetwork(self):
        average_grad_w1=0.1*(self.sumGradient_w1)
        average_grad_b1=0.1*(self.sumGradient_b1)
        average_grad_w2=0.1*(self.sumGradient_w2)
        average_grad_b2=0.1*(self.sumGradient_b2)
        average_grad_w3=0.1*(self.sumGradient_w3)
        average_grad_b3=0.1*(self.sumGradient_b3)
        # print("average grad w3",average_grad_w3.shape)
        self.sumGradient_w1=np.zeros([self.structure[1],self.structure[0]])
        self.sumGradient_b1=np.zeros([self.structure[1],1])
        self.sumGradient_w2=np.zeros([self.structure[2],self.structure[1]])
        self.sumGradient_b2=np.zeros([self.structure[2],1])
        self.sumGradient_w3=np.zeros([self.structure[3],self.structure[2]])
        self.sumGradient_b3=np.zeros([self.structure[3],1])
        self.w1+=-3*(average_grad_w1)
        self.w2+=-3*(average_grad_w2)
        self.w3+=-3*(average_grad_w3)
        self.b1+=-3*(average_grad_b1)
        self.b2+=-3*(average_grad_b2)
        self.b3+=-3*(average_grad_b3)
    
    def evaluateNetwork(self,startingInput,desiredOutput):
        networkOutput=self.feedForward(startingInput)
        indexOfMostActiveNeuron=np.argmax(networkOutput)
        if(indexOfMostActiveNeuron==desiredOutput[0]):
            return True
        else:
            return False

myNetwork=NeuralNetwork([784,128,64,10])
def shuffleTrainingData():
    random.shuffle(training_data)
    return training_data
def batchTrainingData():
    shuffledData=shuffleTrainingData()
    batchedData=[]
    for i in range(0,len(shuffledData),10):
        batchedData.append(shuffledData[i:i+10])
    return batchedData

training_data,testingData=prepareForTraining()
for i in range(50):
    training_set=batchTrainingData()
    for training_batch in training_set:
        for trainingInput,desiredOutput in training_batch:
            myNetwork.feedForward(np.array(trainingInput))
            myNetwork.backwardPropagation(np.array(desiredOutput))
        myNetwork.tuneNetwork()
    correctClassification=0
    for testingInput,desiredTestingOutput in testingData:
        if myNetwork.evaluateNetwork(np.array(testingInput),np.array(desiredTestingOutput)):
            correctClassification+=1
    print("correctClassification:",correctClassification)















