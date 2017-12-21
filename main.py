#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:21:27 2017

@author: sanmin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import simulation

if __name__ == "__main__":
    iterationsLen = 200 
    cordLen = 20
    numAxons = 5
    randomSim = False
    #run the simulation
    sim = simulation.Simulation(length=cordLen, iterations=iterationsLen, axons=numAxons, dt=0.5, random_stim=randomSim)
    
    results = sim.run()
    sim.plot_results()
    axons = results.shape[0] #how many axons in simulation
    length = results.shape[1] #how many compartments for each axon (length of each axon)
    trials = results.shape[2] #number of iterations of the simulation (time)

    #display the results in an animation 
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    #ax.set_xlim(0, length), ax.set_xticks([])
    ax.set_ylim(-1, axons), ax.set_yticks([])
    
    #create the legend
    pain_swatch = mpatches.Patch(color='red', label='Pain Signal')
    sensory_swatch = mpatches.Patch(color='blue', label='Sensory Signal')
    stim_swatch = mpatches.Patch(color='green', label='Agent Stimulus')
    plt.legend(handles=[pain_swatch, sensory_swatch, stim_swatch], loc='lower right')
    
    #create the text on the plot
    text = ax.text(length//2, axons-0.5, "Iteration")
    tspinal = ax.text(-0.5, axons-0.5, "Spinal Cord\n(pain)")
    tneuroma = ax.text(length-1, axons-0.5, "Neuroma\n(parasthesias)")
    taxons = []
    for a in range(axons):
        taxons.append(ax.text(-2, a, "Axon %s" % a))
    
    
    #plot lines on the figure to represent the axons
    for a in range(axons):
        plt.plot([0, length], [a, a], '-k')
    
    #plot the initial state
    x = np.arange(0, length)
    #colors is the colors assigned to each point as a double nested loop [axons x length]
    # (1, 0, 0) for pain, (0, 0, 1) for sensory, (0, 1, 0) for stim, 'none' for no signal
    colors = [[(1, 0, 0) if results[j, i, 0] == -2 else (0, 0, 1) if results[j, i, 0] == -1 else (0, 1, 0) if results[j, i, 0] == 1 else 'none' for i in range(length)] for j in range(axons)]        
    scat = axons*[None]    
    for a in range(axons):  
        y = np.full(shape=[length], fill_value=a)
        scat[a] = ax.scatter(x, y, s=100, lw=5, edgecolors=colors[a], facecolors='none')
    
    
    def update(frame_number):
        colors = [[(1, 0, 0) if results[j, i, frame_number] == -2 else (0, 0, 1) if results[j, i, frame_number] == -1 else (0, 1, 0) if results[j, i, frame_number] == 1 else 'none' for i in range(length)] for j in range(axons)]
        for a in range(axons):
            scat[a].set_edgecolors(colors[a])
        text.set_text('Iteration ' + str(frame_number))

    ani = animation.FuncAnimation(fig, update, frames=trials, repeat=False, interval=200)
    plt.show()