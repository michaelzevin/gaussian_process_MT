# import packages
import numpy as np
import scipy as sp
import pandas as pd
import time
import os
import pdb
import argparse
import itertools
import pickle

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D


# read in arguments
argp = argparse.ArgumentParser()
argp.add_argument("-pd", "--pickle-directory", type=str, default='10_steps', help="Path to pickled output. Default='10_steps'.")
argp.add_argument("-f", "--run-tag", help="Use this as the stem for all file output.")
args = argp.parse_args()


# path to the directory that has all the resampled files you wish to use
pickle_path = 'pickles/' + args.pickle_directory + '/'


# make subdirectory in "plots" for the plots
if not os.path.exists("plots/" + args.run_tag):
    os.makedirs("plots/" + args.run_tag)
pltdir = 'plots/'+args.run_tag+'/'


# read in the pickles
params=[]
pickles={}
for p in os.listdir(pickle_path):
    params.append(p[:-7])
    pickles[p[:-7]] = pickle.load(open(pickle_path+p, "rb"))
inputs = pickles[params[0]]['inputs'] # these will be the same for all output parameters since we specify random seed
full_inputs = pickles[params[0]]['full_inputs']
X_test = pickles[params[0]]['X_test']
X_train = pickles[params[0]]['X_train']
steps = pickles[params[0]]['y_train'].shape[1]


# define function to unnormalize
def unnormalize(norm_vec, min, max):
    return (norm_vec * (max-min) + min)
# do this for everything in X_test, X_train
X_test[:,0] = unnormalize(X_test[:,0],full_inputs['Mbh_init'].min(),full_inputs['Mbh_init'].max())
X_test[:,1] = unnormalize(X_test[:,1],full_inputs['M2_init'].min(),full_inputs['M2_init'].max())
X_test[:,2] = unnormalize(X_test[:,2],full_inputs['P_init'].min(),full_inputs['P_init'].max())
X_test[:,3] = 10**unnormalize(X_test[:,3],np.log10(full_inputs['Z_init']).min(),np.log10(full_inputs['Z_init']).max())
X_train[:,0] = unnormalize(X_train[:,0],full_inputs['Mbh_init'].min(),full_inputs['Mbh_init'].max())
X_train[:,1] = unnormalize(X_train[:,1],full_inputs['M2_init'].min(),full_inputs['M2_init'].max())
X_train[:,2] = unnormalize(X_train[:,2],full_inputs['P_init'].min(),full_inputs['P_init'].max())
X_train[:,3] = 10**unnormalize(X_train[:,3],np.log10(full_inputs['Z_init']).min(),np.log10(full_inputs['Z_init']).max())


# pick random testing point for plotting purposes
t = np.random.randint(0,len(X_test)) # choose random testing track to plot, or specify number
print 'Testing point properties:'
print '   Mbh_init : %f' % X_test[t,0]
print '   M2_init : %f' % X_test[t,1]
print '   P_init : %f' % X_test[t,2]
print '   Z_init : %f' % X_test[t,3]


### Plot initial condition of entire dataset, and training vs testing set ###
fig=plt.figure(figsize = (15,12), facecolor = 'white')
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel('$Black\ Hole\ Mass\ (M_{\odot})$', rotation=0, labelpad=20, size=12)
ax.set_ylabel('$Companion\ Mass\ (M_{\odot})$', rotation=0, labelpad=20, size=12)
ax.set_xlabel('$Log\ Period\ (s)$', rotation=0, labelpad=20, size=12)

pts = ax.scatter(np.log10(X_train[:,2]), X_train[:,1], X_train[:,0], zdir='z', cmap='viridis', c=X_train[:,3], vmin=inputs["Z_init"].min(), vmax=inputs["Z_init"].max(), marker='.', s=10, label='training tracks')
ax.scatter(np.log10(X_test[:,2]), X_test[:,1], X_test[:,0], zdir='z', cmap='viridis', c=X_test[:,3], vmin=inputs["Z_init"].min(), vmax=inputs["Z_init"].max(), marker='*', s=20, label='testing tracks')
ax.scatter(np.log10(X_test[t,2]), X_test[t,1], X_test[t,0], zdir='z', cmap='viridis', c=X_test[t,3], vmin=inputs["Z_init"].min(), vmax=inputs["Z_init"].max(), marker='*', s=200, label='plotted point')
fig.colorbar(pts)
plt.legend()

plt.tight_layout()
fname = 'param_space.png'
if args.run_tag:
    fname = pltdir + fname
plt.savefig(fname)



### Plot evolution comparison for a given track ###
f, axs = plt.subplots(nrows = len(params), sharex=True, figsize=(12,1.5*len(params)))
for idx, p in enumerate(params):
    # setup axes
    if idx==len(params)-1:
        axs[idx].set_xlabel('Resampled Step')
    axs[idx].set_ylabel(p)
    if idx==0:
        axs[idx].set_title('Interpolation for Testing Track Mbh: '+str(X_test[:,0])+', M2: '+str(X_test[:,1])+', P: '+str(X_test[:,2])+', Z: '+str(X_test[:,3]))
    axs[idx].set_xlim(0-steps/10, steps+steps/10) # add some buffer to the plot

    # do plotting
    param = pickles[p]
    axs[idx].plot(np.linspace(0,steps,steps), param['y_test'][t,:], 'k', linewidth=1, alpha=0.5, label='actual evolution')
    axs[idx].plot(np.linspace(0,steps,steps), param['linear'][t,:], 'g.', linewidth=3, alpha=0.5, label='linear interpolated evolution')
    axs[idx].plot(np.linspace(0,steps,steps), param['GP'][t,:], 'b.', linewidth=3, alpha=0.5, label='GP interpolated evolution')
    axs[idx].fill_between(np.linspace(0,steps,steps), param['GP'][t,:]-param['error'][t,:], param['GP'][t,:]+param['error'][t,:], alpha=0.05, label='GP error')
plt.legend()
fname = 'test_evolution.png'
if args.run_tag:
    fname = pltdir + fname
plt.tight_layout()
plt.savefig(fname)


### Plot the average error for each output parameter, scaled by the range of the output in question ###

def mean_exp_error(exp,act):
    exp_err = np.abs((exp-act)/act)
    return exp_err.mean()*100, exp_err.std()*100

f, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
# setup axes
axs.set_xlabel('Parameter')
axs.set_ylabel('Percent Error (%)')
axs.set_title('Average Percent Error for Interpolation Methods')
axs.set_xlim(-1, len(params)) # add some buffer to the plot
axs.set_ylim(0,200)
plt.xticks(range(len(params)), params)


# do plotting
for idx, p in enumerate(params):
    print idx, p
    param = pickles[p]
    GP_err, GP_std = mean_exp_error(param['GP'],param['y_test'])
    lin_err, lin_std = mean_exp_error(param['linear'],param['y_test'])
    axs.scatter(idx-0.2, GP_err, c='b', marker='*')
    axs.scatter(idx+0.2, lin_err, c='g', marker='*')
    axs.errorbar(idx-0.2, GP_err, c='b', yerr=GP_std, label='average GP error')
    axs.errorbar(idx+0.2, lin_err, c='g', yerr=lin_std, label='average linear error')
    if idx==0:
        plt.legend(loc='upper right')

fname = 'global_err.png'
if args.run_tag:
    fname = pltdir + fname
plt.tight_layout()
plt.savefig(fname)


### See how GP-predicted error corellates to actual error ###

def abs_err(exp,act):
    err = np.abs(exp-act)
    return err

f, axs = plt.subplots(nrows=len(params), ncols=1, figsize=(9,3*len(params)))

# do plotting
for idx, p in enumerate(params):
    # setup axes
    axs[idx].set_title(p)
    axs[idx].set_ylabel('Actual error')
    if idx == len(params)-1:
        axs[idx].set_xlabel('GP error')
    param = pickles[p]
    actual_error = abs_err(param['GP'],param['y_test'])
    axs[idx].scatter(param['error'].flatten(), actual_error.flatten(), c='k', marker='.')

fname = 'error_comp.png'
if args.run_tag:
    fname = pltdir + fname
plt.tight_layout()
plt.savefig(fname)
