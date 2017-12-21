#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:25:47 2017

@author: sanmin
"""

import numpy as np
import matplotlib.pyplot as plt
import axon

class Simulation(object):
    def __init__(self, length, iterations, axons=1, dt=1, random_stim=False):
        '''
        Length - int, number of compartments in axon
        iterations - int, number of steps to run the simulation for (run in for loop for 0 to iterations+1)
        dt - float, time (s) of each iteration
        '''
        self.tol = 10**(-5)
        self.const_freq = iterations // 20 #so we have an average of 30 firings for each axon
        self.iterations = iterations
        self.current_iteration = 0
        self.dt = dt
        self.t = np.linspace(0, self.iterations*self.dt+self.dt, self.iterations+1)
        self.num_axons = axons
        self.random_stim = random_stim #If true, just do random stim instead of agent deciding
        
        # create a list of axons
        self.myAxons = [axon.Axon(length) for i in range(self.num_axons)]
        
        #create a stack of axons - call axon_record[axon, compartment_in_axon]
        self.axon_record = [ax.get_axon()[0] for ax in self.myAxons] #save the axon for each iteration
        self.axon_record = np.stack(self.axon_record, axis=0)   
        
        # set sensory signals (at random frequencies for each axon)
        self.sensory_freqs = [np.random.randint(self.const_freq-5, self.const_freq+5) for i in range(self.num_axons)]
        self.sensory_signals = [self.set_sensory_signal(freq)[0] for freq in self.sensory_freqs] 
        self.sensory_signals = np.stack(self.sensory_signals, axis=0)
        
        #choose one axon that is the pain axon
        self.pain_axon = np.random.randint(0, self.num_axons)
        
        #------------- Agent performance stuff -----------
        self.total_parasthesias = np.zeros(shape=(self.iterations+1, 1))
        self.total_pain = np.zeros(shape=(self.iterations+1, 1))
        
        #-------------- Agent stuff ------------------
        self.pain_t = [] #time of each pain
        self.sensory_t = [[] for i in range(self.num_axons)] #the time that each sensory signal reaches the end
        self.para_t = [] #time of each parasthesia
        self.stim_t = [0] #time of each stim
        self.pain_iterations = 0 #the average number of iterations between pain signals (frequency of pain signals)
        self.can_stimulate = False #set to true once the agent has recorded enough information about the environment     
        self.agent_axon = self.pain_axon #the axon that the agent thinks is sending the pain signal        
        
    def set_sensory_signal(self, freq):
        #create a vector of 0's and 1's that is the length of number of iterations
        # 0 for no pain sent, 1 for pain sent
        sensory = np.zeros(shape=(1,self.iterations+1))
        
        #constant frequency
        i = 0 + np.random.randint(freq - 1, freq + 2) #random frequency 
        while i < self.iterations+1:
            sensory[0, i] = 1
            i += np.random.randint(freq - 1, freq + 2) #could be random
        
        return sensory
    
    
    def send_pain(self):
        '''
        Return true if it's time to send pain signal, false otherwise
        '''
        if self.sensory_signals[self.pain_axon, self.current_iteration]:
            return True
        else:
            return False
    
    
    def send_stim(self):
        '''
        AGENT
        Return true if it's time to send stim signal, false otherwise
        '''
        #iterations_between_last_pain = self.current_iteration - self.pain_t[-1]
        iterations_between_last_stim = self.current_iteration - self.stim_t[-1]
        #print('last stim ' + str(iterations_between_last_stim))
        if self.pain_iterations == 0:
            return False
        if  iterations_between_last_stim >= self.pain_iterations:
#            print(self.pain_iterations)
            return True
        else:
            return False
    
    def record_pain_frequency(self):
        #calculate the average iterations between pain signals
        if len(self.pain_t) > 1: #ignore the first pain signal since it takes longer to reach end
            s = 0            
            for i in range(1, len(self.pain_t)):
                if i > 0:                
                    s += self.pain_t[i] - self.pain_t[i-1]
            self.pain_iterations = np.floor(float(s) / (len(self.pain_t)-1))
            
        #find the closest axon that has sensory signals that match the pain signals
        diff = self.num_axons * [0]
        for a in range(self.num_axons):
            for t in range(np.minimum(len(self.pain_t), len(self.sensory_t[a]))):
                diff[a] = diff[a] + np.abs(self.pain_t[t] - self.sensory_t[a][t])
        self.agent_axon = np.argmin(diff)
        
#        print '-------------------- agent----------------'
#        print self.pain_t
#        print self.sensory_t
#        print diff
#        print 'pain freq: %s pain axon: %s' % (self.pain_iterations, self.agent_axon)
    
    
    def run(self): 
        new_axon_records = [self.axon_record]
        for t in range(1, self.iterations+1): 
            self.current_iteration = t
            
            #send a sensory signal according to the set self.sensory_signals
            axons = np.argwhere(self.sensory_signals[:, self.current_iteration])              
            for a in axons.tolist():
                self.myAxons[a[0]].send_pain_signal()
            
            #Let the agent do stuff here (send a stimulus or not)
            if self.random_stim:
                #randomly stimulate 
                if np.random.randint(0, 2) == 0:
                    self.myAxons[np.random.randint(0,self.num_axons)].send_stim_signal()
            else: #let the agent decide what to do
                if len(self.pain_t) < 7:            
                    self.record_pain_frequency() 
                elif self.send_stim():
                    #print('freq ' + str(self.pain_iterations))
                    self.stim_t.append(t)
                    self.myAxons[self.agent_axon].send_stim_signal()

            #update the environment
            sensory_para = [ax.step() for ax in self.myAxons]
            sensory = [sensory_para[i][1] for i in range(len(sensory_para))]
            para = [sensory_para[i][0] for i in range(len(sensory_para))]
            
            #update the charts
            self.total_parasthesias[t, 0] = para.count(1)
            self.total_pain[t, 0] = sensory[self.pain_axon]
            
            #update the memory of the state for the agent
            if para.count(1) > 0:
                self.para_t.append(t)
            if sensory[self.pain_axon] > 0:
                self.pain_t.append(t)
#            print '--------- sensory ----------'
#            print sensory
            for a in range(len(sensory)):
                if sensory[a] > 0:
                    self.sensory_t[a].append(t)
#                    print self.sensory_t

            #save the axon
            new_axon_record = [ax.get_axon()[0] for ax in self.myAxons]
            new_axon_record = np.stack(new_axon_record, axis=0) 
            #turn the pain axon signals to -2
            new_axon_record[self.pain_axon, np.argwhere(new_axon_record[self.pain_axon, :] == -1)] = -2
            new_axon_records.append(new_axon_record)
            
#            print '-------------------------- iteration ---------------'   
#            print new_axon_record
            
        self.axon_record = np.dstack(new_axon_records)
        
        return self.axon_record        
    
    def plot_results(self):
        #plot the pain and parasthesia instances versus time
        plt.figure(1)
        plt.plot(np.arange(0, self.iterations+1), self.total_pain[:,0], 'o')
        plt.ylim(-1, 2)
        plt.xlabel('Iteration')
        plt.ylabel('Total number of pain')
        plt.title('Pain')
        
        plt.figure(2)
        plt.plot(np.arange(0, self.iterations+1), self.total_parasthesias[:,0], 'o')
        plt.ylim(-1, 2)
        plt.xlabel('Iteration')
        plt.ylabel('Total number of parasthesias')
        plt.title('Parasthesias')
        
if __name__ == "__main__":
    sim = Simulation(10, 200, 5, random_stim=True)
    results = sim.run()
    sim.plot_results()
#    print results.shape
    

    
    
    
