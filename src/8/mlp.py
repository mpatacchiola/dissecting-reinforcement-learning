#!/usr/bin/env python

#MIT License
#Copyright (c) 2017 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#Simple class for a Multi-Layer Perceptron (MLP)
#It is based on numpy calls and completely vectorized

import numpy as np

class MLP():

    def __init__(self, tot_inputs, tot_hidden, tot_outputs, activation="sigmoid"):
        '''Init an MLP object

        Defines the matrices associated with the MLP.
        @param: tot_inputs
        @param: tot_hidden
        @param: tot_outputs
        '''
        import numpy as np
        
        if(activation!="sigmoid" and  activation!="tanh"):
            raise ValueError("[ERROR] The activation function " 
                             + str(activation) + " does not exist!")
        else:
            self.activation = activation
        self.tot_inputs = tot_inputs
        self.W1 = np.random.normal(0.0, 0.1, (tot_inputs+1, tot_hidden))
        self.W2 = np.random.normal(0.0, 0.1, (tot_hidden+1, tot_outputs))
        self.tot_outputs = tot_outputs

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _tanh(self, z):
        return np.tanh(z)
                
    def forward(self, x, verbose=False):
        '''Forward pass in the neural network

        Forward pass via dot product
        @param x: the input vector (must have shape==tot_inputs)
        @param verbose: print h, W1, W2, and y
        @return: the output of the network
        '''
        if(x.shape[0]!=self.tot_inputs): raise ValueError("[ERROR] The size of x is wrong!")
        self.x = np.hstack([x, np.array([1.0])]) #add the bias unit
        self.z1 = np.dot(self.x, self.W1)
        if(self.activation=="sigmoid"):
            self.h = self._sigmoid(self.z1)
        elif(self.activation=="tanh"):
            self.h = self._tanh(self.z1)
        self.h = np.hstack([self.h, np.array([1.0])]) #add the bias unit
        self.z2 = np.dot(self.h, self.W2) #shape [tot_outputs]
        if(self.activation=="sigmoid"):
            self.y = self._sigmoid(self.z2)
        elif(self.activation=="tanh"):
            self.y = self._tanh(self.z2)
        if(verbose): print("h: " + str(self.h))
        if(verbose): print("W1: " + str(self.W1))
        if(verbose): print("W2: " + str(self.W2))
        if(verbose): print("y: " + str(self.y))
        return self.y

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1.0 - self._sigmoid(z))

    def _tanh_derivative(self, z):
        return 1 - np.square(np.tanh(z))
       
    def backward(self, target, learning_rate=0.01, verbose=False):
        '''Backward pass in the network
  
        @param target: the value of taget vector (same shape of output)
        @param learning_rate: the learning rate (default 0.01)
        @param verbose: print the gradient vectors
        @return: two matrices dW1 and dW2
        '''
        if(self.y.shape[0]!=target.shape[0]): raise ValueError("[ERROR] The size of target is wrong!")
        #gathering all the partial derivatives
        dE_dy = -(target - self.y) #shape [tot_outputs]
        if(self.activation=="sigmoid"):
            dy_dz2 = self._sigmoid_derivative(self.z2)
        elif(self.activation=="tanh"):
            dy_dz2 = self._tanh_derivative(self.z2)
        dz2_dW2 = self.h        
        dz2_dh = self.W2
        if(self.activation=="sigmoid"):
            dh_dz1 = self._sigmoid_derivative(self.z1) #shape [tot_hidden]
        elif(self.activation=="tanh"):
            dh_dz1 = self._tanh_derivative(self.z1) #shape [tot_hidden]      
        dz1_dW1 = self.x
        #gathering the gradient vector of W2
        #shape = [tot_outputs] * [tot_outputs] * [tot_hidden]
        dE_dW2 = np.dot(np.expand_dims(dE_dy * dy_dz2, axis=1), 
                        np.expand_dims(dz2_dW2, axis=0)).T
        #gathering the gradient of W1
        dE_dW1 = (dE_dy * dy_dz2) #shape = [tot_outputs] * [tot_outputs]
        dE_dW1 = np.dot(dE_dW1, dz2_dh.T)[0:-1] * dh_dz1 #bias removed: [tot_outputs] * [tot_hidden+1, tot_outputs]
        dE_dW1 = np.dot(np.expand_dims(dE_dW1,axis=1), 
                        np.expand_dims(dz1_dW1,axis=0)).T #[tot_hidden] * [tot_inputs+1]
        if(verbose): print("dE_dW1: " + str(dE_dW1))
        if(verbose): print("dE_dW2: " + str(dE_dW2))
        #update the weights
        self.W2 = self.W2 - (learning_rate * dE_dW2) 
        self.W1 = self.W1 - (learning_rate * dE_dW1)

    def train(self, x, target, learning_rate=0.1):
        '''train the network
  
        @param x: the input vector
        @param target: the target value vector
        @param learning_rate: the learning rate (default 0.01)
        @param verbose: print the gradient vectors
        @return: the error RMSE
        '''
        y = self.forward(x)
        self.backward(target, learning_rate)
        error = 0.5 * (target - y)**2
        return error

#def main():
#    my_mlp = MLP(tot_inputs=2, tot_hidden=3, tot_outputs=2, activation="tanh")
#    
#    for i in range(10000):
#       my_mlp.forward(np.array([1.0, 1.0]), verbose=True)
#       print("[step " + str(i) + "] y=" + str(my_mlp.y))
#       my_mlp.backward(target=np.array([-0.75, 0.25]), learning_rate=0.1, verbose=True)
#       print
'''
       
def main():
    my_mlp = MLP(tot_inputs=2, tot_hidden=2, tot_outputs=2)
    my_mlp.W1 = np.array([[0.15, 0.20],
                          [0.25, 0.30],
                          [0.35, 0.35]], dtype=np.float32)
                          
    my_mlp.W2 = np.array([[0.40, 0.45],
                          [0.50, 0.55],
                          [0.60, 0.60]], dtype=np.float32)
                          
    y = my_mlp.forward(np.array([0.05, 0.1]), verbose=True)
    print(y)
    
    my_mlp.backward(target=np.array([0.01, 0.99]), learning_rate=0.1, verbose=True)
'''   
       
       

if __name__ == "__main__":
    main()
