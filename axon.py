"""
Created on Wed Nov  8 12:57:55 2017

@author: diana
"""

import numpy as np

# Axons' model
class Axon(object):
    def __init__(self, length):
        '''
        Length - int, number of compartments in axon
        '''
        self.n = length + 2 # +2 for transmitting side
        self.axon = np.zeros(shape=(1,self.n))  # 1 by self.n
        self.signals = [] #stores index of all the signals        

    def send_pain_signal(self):
        '''
        Sends a pain signal - pain signals travel from axon[end] to axon[0] (to the left so -1)
        '''
        self.axon[0, -1] = -1
        self.signals.append(int(self.n-1))

        
        
    def send_stim_signal(self):
        '''
        Sends a stimulation signal - signals travel from axon[0] to axon[end] (to the right so +1)
        '''        
        self.axon[0, 0] = 1
        self.signals.append(0)
    

    def step(self):
        '''
        Move all the signals in the axon for one time step
        
        Output: number of parasthesias, number of pain
        '''
        output = len(self.signals)*[0]
        new_signals = []
        for i in range(len(self.signals)):
            index = self.signals[i]
            new_index = int(index + self.axon[0, index]) #self.axon[index] will be -1 if signal moving left, +1 if right

            if new_index >= self.n or new_index < 0: #signal out of range of axon
                #print('signal exited the axon')                
                output[i] = self.axon[0, index]
            elif self.axon[0, new_index] != 0: #collision
                #print('collision')                
                self.axon[0, new_index] = 0
            else: #normal move
                self.axon[0, new_index] = self.axon[0, index] + self.axon[0, new_index]
                new_signals.append(new_index) 
            #remove the signal from the old spot
            self.axon[0, index] = 0
            
        self.signals = new_signals
        
        return output.count(1), output.count(-1)

    def get_axon(self):
        '''
        Return the axon vector
        '''
        return self.axon
