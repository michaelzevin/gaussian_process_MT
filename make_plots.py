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
argp.add_argument("-pd", "--pickle-directory", type=str, default='10_steps', help="Name of the pickle directory. Default='10_steps'.")
argp.add_argument("-f", "--run-tag", help="Use this as the stem for all file output.")
args = argp.parse_args()


# path to the directory that has all the resampled files you wish to use
basepath = os.getcwd()
pickle_path = basepath + '/pickles/' + args.pickle_directory + '/'


# make subdirectory in "plots" for the plots
if args.run_tag:
    if not os.path.exists("plots/" + args.run_tag):
        os.makedirs("plots/" + args.run_tag)
    pltdir = 'plots/'+args.run_tag+'/'
else:
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    pltdir = 'plots/'


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


# pick random testing point for plotting purposes (for testMT, it is the only available track)
t = np.random.randint(0,len(X_test)) # choose random testing track to plot, or specify number
print 'Testing point properties:'
print '   Mbh_init : %f' % X_test.iloc[t]['Mbh_init']
print '   M2_init : %f' % X_test.iloc[t]['M2_init']
print '   P_init : %f' % X_test.iloc[t]['P_init']
print '   Z_init : %f' % X_test.iloc[t]['Z_init']


### Plot initial condition of entire dataset, and training vs testing set ###
fig=plt.figure(figsize = (15,12), facecolor = 'white')
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel('$Black\ Hole\ Mass\ (M_{\odot})$', rotation=0, labelpad=20, size=12)
ax.set_ylabel('$Companion\ Mass\ (M_{\odot})$', rotation=0, labelpad=20, size=12)
ax.set_xlabel('$Log\ Period\ (s)$', rotation=0, labelpad=20, size=12)

pts = ax.scatter(np.log10(np.array(X_train['P_init'])), np.array(X_train['M2_init']), np.array(X_train['Mbh_init']), zdir='z', cmap='viridis', c=np.array(X_train['Z_init']), vmin=inputs["Z_init"].min(), vmax=inputs["Z_init"].max(), marker='.', s=10, label='training tracks')
ax.scatter(np.log10(np.array(X_test['P_init'])), np.array(X_test['M2_init']), np.array(X_test['Mbh_init']), zdir='z', cmap='viridis', c=np.array(X_test['Z_init']), vmin=inputs["Z_init"].min(), vmax=inputs["Z_init"].max(), marker='*', s=20, label='testing tracks')
ax.scatter(np.log10(X_test['P_init'].iloc[t]), X_test['M2_init'].iloc[t], X_test['Mbh_init'].iloc[t], zdir='z', cmap='viridis', c=X_test['Z_init'].iloc[t], vmin=inputs["Z_init"].min(), vmax=inputs["Z_init"].max(), marker='*', s=200, label='plotted point')
fig.colorbar(pts)
plt.legend()

plt.tight_layout()
fname = pltdir + 'param_space.png'
plt.savefig(fname)



### Plot evolution comparison for a given track ###
f, axs = plt.subplots(nrows = len(params), sharex=True, figsize=(12,1.5*len(params)))
for idx, p in enumerate(params):
    # setup axes
    if idx==len(params)-1:
        axs[idx].set_xlabel('Resampled Step')
    axs[idx].set_ylabel(p)
    if idx==0:
        axs[idx].set_title('Interpolation for Testing Track M2: '+str(X_test['M2_init'].iloc[t])+', Mbh: '+str(X_test['Mbh_init'].iloc[t])+', P: '+str(X_test['P_init'].iloc[t])+', Z: '+str(X_test['Z_init'].iloc[t]))
    axs[idx].set_xlim(0-steps/10, steps+steps/10)   # add some buffer to the plot

    # do plotting
    param = pickles[p]
    axs[idx].plot(np.linspace(0,steps,steps), param['y_test'][t,:], 'k', linewidth=1, alpha=0.5, label='actual evolution')
    axs[idx].plot(np.linspace(0,steps,steps), param['linear'][t,:], 'g', linewidth=0.5, alpha=0.5, label='linear interpolated evolution')
    axs[idx].plot(np.linspace(0,steps,steps), param['GP'][t,:], 'b', linewidth=0.5, alpha=0.5, label='GP interpolated evolution')
    #axs[idx].fill_between(np.linspace(0,steps,steps), param['GP'][t,:]-param['error'][t,:], param['GP'][t,:]+param['error'][t,:], alpha=0.05, label='GP error')
plt.legend()
fname = pltdir + 'test_evolution.png'
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
    # FIXME it would be good to return these values below for comparison...
    GP_err, GP_std = mean_exp_error(param['GP'],param['y_test'])
    lin_err, lin_std = mean_exp_error(param['linear'],param['y_test'])
    axs.scatter(idx-0.2, GP_err, c='b', marker='*')
    axs.scatter(idx+0.2, lin_err, c='g', marker='*')
    axs.errorbar(idx-0.2, GP_err, c='b', yerr=GP_std, label='average GP error')
    axs.errorbar(idx+0.2, lin_err, c='g', yerr=lin_std, label='average linear error')
    if idx==0:
        plt.legend(loc='upper right')

fname = pltdir + 'global_err.png'
plt.tight_layout()
plt.savefig(fname)


### See how GP-predicted error corellates to actual error ###
# FIXME this isn't working...

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

fname = pltdir + 'error_comp.png'
plt.tight_layout()
plt.savefig(fname)
