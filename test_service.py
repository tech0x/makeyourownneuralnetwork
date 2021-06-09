#!/usr/bin/python3

import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
import imageio
import base64
import json
from skimage.transform import resize

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #self.wih = (np.random.rand(self.hnodes, self.inodes)-0.5)
        #self.who = (np.random.rand(self.onodes, self.hnodes)-0.5)
        
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
                
        self.lr = learningrate        

        self.activation_function = lambda x: sc.expit(x)
        pass
        
    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        
        hidden_inputs = self.wih@inputs
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = self.who@hidden_outputs
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = self.who.T@output_errors
        self.who += self.lr * np.dot((output_errors*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)), np.transpose(inputs))
        pass
        
    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        
        hidden_inputs = self.wih@inputs
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = self.who@hidden_outputs
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 450
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

print("Start learning")
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = (np.asfarray(all_values[1:])/255*0.99)+0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
    pass

print("End learning")
test_data_file = open("mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:])/255*0.99)+0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print("Performance=", scorecard_array.sum()/scorecard_array.size)

def application(env, start_response):
    start_response('200 OK', [('Content-Type','text/html')])
    print(env['QUERY_STRING'])
    if len(env['QUERY_STRING']) > 300:
        f = open("/tmp/temp.png", "wb")
        f.write(base64.b64decode(env['QUERY_STRING']))
        f.close()
        img_array = imageio.imread("/tmp/temp.png", as_gray=True)
        img_array = resize(img_array, (28,28))
        img_data = 255.0 - img_array.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01
        print("min = ", np.min(img_data))
        print("max = ", np.max(img_data))

        outputs = n.query(img_data)
        print(np.argmax(outputs))
        print(outputs)

        #print(env, start_response)
        return [b"Result: %d "%np.argmax(outputs)]


