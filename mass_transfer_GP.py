# import packages
import numpy as np
import scipy as sp
import pandas as pd
from scipy.interpolate import griddata
import argparse
import time
import os
import pdb
import multiprocessing
import itertools
import pickle

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn import gaussian_process
GP = gaussian_process.GaussianProcess
GPR = gaussian_process.GaussianProcessRegressor
GPC = gaussian_process.GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern as matern, ExpSineSquared as ess
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
print "Scikit-learn version: %s" % (sklearn.__version__)


# read in arguments
argp = argparse.ArgumentParser()
argp.add_argument("-rp", "--resamp-path", type=str, default='/projects/b1011/mzevin/gaussian_process/data/fine_grid/resampled/', help="Path to resampled tracks directory. Default='/projects/b1011/mzevin/gaussian_process/data/fine_grid/resampled/'.")
argp.add_argument("-r", "--resamp", type=str, default='all_l2_10', help="Resampled tracks you wish to use for interpolation. Default='all_l2_10'.")
argp.add_argument("-p", "--parameter", type=str, help="Parameter you wish to interpolate. Parameters of interest include: star_1_mass, star_2_mass, log_Teff, log_L, log_R, period_days, age, log_dt, log_abs_mdot, binary_separation, lg_mtransfer_rate, lg_mstar_dot_1, lg_mstar_dot_2, etc. Note: star_1 = companion, star_2 = black hole.")
argp.add_argument("-t", "--test-set", type=float, default=0.2, help="Fraction of the total set that is held out for testing (i.e., the GP will be trained on the (1-t) datapoints). Default = 0.2.")
argp.add_argument("-rs", "--random-seed", type=int, help="Use this for reproducible output.")
argp.add_argument("-f", "--run-tag", help="Use this as the stem for all file output.")
argp.add_argument("-S", "--save-pickle", action="store_true", help="Save the GP model as a pickle. Default is off.")
argp.add_argument("-P", "--make-plots", action="store_true", help="Makes and saves plots. Default is off.")
args = argp.parse_args()


# argument handlingi
make_plots = args.make_plots
save_pickle = args.save_pickle


# path to the directory that has all the resampled files you wish to use
resamp_path = args.resamp_path + args.resamp + '/'


# dict of parameter names
param_names=['log_dt','log_abs_mdot','he_core_mass','c_core_mass','o_core_mass','mass_conv_core','log_LH','log_LHe','log_LZ','log_Lnuc','log_Teff','log_L','log_R','log_g','surf_avg_omega','surf_avg_omega_div_omega_crit','center_h1','center_he4','center_c12','center_o16','surface_c12','surface_o16','total_mass_h1','total_mass_he4','log_center_P','log_center_Rho','log_center_T','model_number','age','period_days','binary_separation','v_orb_1','v_orb_2','star_1_radius','rl_1','rl_2','rl_relative_overflow_1','rl_relative_overflow_2','star_1_mass','star_2_mass','lg_mtransfer_rate','lg_mstar_dot_1','lg_mstar_dot_2','lg_system_mdot_1','lg_system_mdot_2','lg_wind_mdot_1','lg_wind_mdot_2','xfer_fraction','J_orb','Jdot','jdot_mb','jdot_gr','jdot_ml','jdot_ls','jdot_missing_wind','extra_jdot','donor_index','point_mass_index','r_tau100','r_tau1000','t_dynamical_tau100','t_dynamical_tau1000','m_dynamical_tau100','m_dynamical_tau100_1','t_thermal_tau1000','t_thermal_tau1000_1','m_thermal_tau100','m_thermal_tau1000']

# find the index of the parameter of interest
param_idx = param_names.index(args.parameter)


# read in inputs and outputs
inputs = []
outputs=[]
for file in os.listdir(resamp_path):
    x = np.load(resamp_path + file)
    inputs.append((np.float(x['Mbh_init']),np.float(x['M2_init']),np.float(x['P_init']),np.float(x['Z_init'])))
    outputs.append(x['x_resamp'][:,param_idx])

# reshape as arrays
inputs = np.reshape(inputs,[len(inputs),4]) # 4 inputs
outputs = np.asarray(outputs)
resamp_len = outputs.shape[1]

# store inputs as a dataframe
inputs_df = pd.DataFrame({"Mbh_init": inputs[:,0], "M2_init": inputs[:,1], "P_init": inputs[:,2], "Z_init": inputs[:,3]})

print 'This grid contains:'
print 'Mbh_init: %f - %f' % (inputs_df["Mbh_init"].min(), inputs_df["Mbh_init"].max())
print 'M2_init: %f - %f' % (inputs_df["M2_init"].min(), inputs_df["M2_init"].max())
print 'P_init: %f - %f' % (inputs_df["P_init"].min(), inputs_df["P_init"].max())
print 'Z_init: %f - %f' % (inputs_df["Z_init"].min(), inputs_df["Z_init"].max())

# normalize the inputs (all values should be positive)
inputs = pd.DataFrame({"Mbh_init": inputs_df['Mbh_init']/inputs_df['Mbh_init'].max(),"M2_init": inputs_df['M2_init']/inputs_df['M2_init'].max(),"P_init": inputs_df['P_init']/inputs_df['P_init'].max(),"Z_init": inputs_df['Z_init']/inputs_df['Z_init'].max()})
inputs = np.array(inputs)


# plot the parameter space coverage
if make_plots:
    fig=plt.figure(figsize = (12,8), facecolor = 'white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel('$Black\ Hole\ Mass\ (M_{\odot})$', rotation=0, labelpad=20, size=12)
    ax.set_ylabel('$Companion\ Mass\ (M_{\odot})$', rotation=0, labelpad=20, size=12)
    ax.set_xlabel('$Log\ Period\ (s)$', rotation=0, labelpad=20, size=12)

    log_P = np.log10(inputs_df['P_init'])
    log_Z = np.log10(inputs_df['Z_init'])
    norm_log_Z = (log_Z-log_Z.min())/(log_Z.max()-log_Z.min())

    pts = ax.scatter(log_P, inputs_df['M2_init'], inputs_df['Mbh_init'], zdir='z', s=5, cmap='viridis', c=norm_log_Z, label='simulated tracks')
    fig.colorbar(pts)
    plt.tight_layout()
    plt.legend()
    fname = 'init_condit.png'
    if args.run_tag:
        fname = args.run_tag + '_' + fname
    plt.savefig(fname)


# split dataset into training and testing sets
if args.random_seed:
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(inputs, outputs, test_size=args.test_set, random_state=args.random_seed)
else:
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(inputs, outputs, test_size=args.test_set)
pdb.set_trace()


# add an extra dimension of 'step' to the normalized inputs and flatten
step_space = np.linspace(0,resamp_len,resamp_len)/resamp_len
X_train = np.array(list(np.append(x,y) for x in X_train_orig for y in step_space))
X_test = np.array(list(np.append(x,y) for x in X_test_orig for y in step_space))
y_train = y_train_orig.flatten()
y_test = y_test_orig.flatten()


### Define GP and auxiliary functions ###

def GPR_scikit(X_train, y_train, X_test):
    X = np.atleast_2d(X_train)
    y = np.atleast_2d(y_train).T
    X_pred = np.atleast_2d(X_test)

    # get min and max of outputs for the constant kernel's bounds #FIXME is this necessary?
    c_min = np.abs(y_train).min()
    c_max = np.abs(y_train).max()

    kernel = C(1e0,(c_min,c_max))*RBF(length_scale=1e-1, length_scale_bounds=(1e-6,1e0))
    kernel = RBF(length_scale=1e-1, length_scale_bounds=(1e-3,1e0))

    yerr = 1e-3
    gp = GPR(kernel=kernel, n_restarts_optimizer=9, normalize_y = True)
    gp.fit(X, y)

    y_pred, sigma = gp.predict(X_pred, return_std=True)
    y_pred = np.reshape(y_pred,len(y_pred))

    return y_pred, sigma

def linear_interp(X_train, y_train, X_test):
    value = sp.interpolate.griddata(X_train, y_train, X_test, method='linear', fill_value=0.0)
    return value


# Specify number of cores to run on FIXME these functions are depreciated
# num_cores = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(num_cores-2)
# notation for parallelizing: data[0]: X_train, data[1]: y_train, data[2]: X_test
def GPR_scikit_multi(data):

    X = np.atleast_2d(data[0])
    y = np.atleast_2d(data[1]).T # need to transpose to make dimensions match since y is 1D

    # get min and max of outputs for the constant kernel's bounds
    c_min = np.abs(data[1]).min()
    c_max = np.abs(data[1]).max()

    kernel = C(1e0,(c_min,c_max))*RBF(length_scale = [1e-1,1e-1,1e-1,1e-1], length_scale_bounds=[(1e-6,1e0),(1e-6,1e0),(1e-6,1e0),(1e-6,1e0)])

    yerr = 1e-8

    # choose between old and new GP package
    gp = GPR(kernel=kernel, n_restarts_optimizer=9, normalize_y = False) # new

    gp.fit(X, y)

    X_pred = np.atleast_2d(data[2])

    y_pred, sigma = gp.predict(X_pred, return_std=True)
    y_pred = np.reshape(y_pred,len(y_pred))

    return y_pred, sigma


def linear_interp_multi(data):
    value = sp.interpolate.griddata(data[0],data[1],data[2], method='linear', fill_value=0.0)
    return value


def center(y):
    means=[]
    y_cen = np.empty(np.shape(y))
    for i in xrange(len(y.T)):
        means.append(y[:,i].mean())
        y_cen[:,i] = y[:,i]-y[:,i].mean()
    return y_cen, means


def uncenter(y_cen, means):
    for i in xrange(len(y_cen.T)):
        y[:,i] = y_cen[:,i]+means[i]
    return y


# define PCA
num_comps=10 # adjust number of components
pca = PCA(n_components=num_comps)

def PCA_to_vals(y):
    return pca.inverse_transform(y)



# do the GP inteprolation
start_time = time.time()   # start the clock
GP_pred, sigma = GPR_scikit(X_train, y_train, X_test)
elapsed = time.time() - start_time   # See how long it took 
print '      Done with GP interpolation for %s...it only took %f seconds!' % (args.parameter,elapsed)


# do the linear inteprolation
start_time = time.time()   # start the clock
lin_pred = linear_interp(X_train, y_train, X_test)
elapsed = time.time() - start_time   # See how long it took
print '      Done with linear interpolation for %s...it only took %f seconds!' % (args.parameter,elapsed)


# reshape interpolations as (tracks,steps) for convenience
GP_pred = np.reshape(GP_pred, (y_test_orig.shape[0],y_test_orig.shape[1]), order='C')
sigma = np.reshape(sigma, (y_test_orig.shape[0],y_test_orig.shape[1]), order='C')
lin_pred = np.reshape(lin_pred, (y_test_orig.shape[0],y_test_orig.shape[1]), order='C')


# save pickle
if save_pickle:
    data = {"inputs": inputs_df, "X_test": X_test_orig, "y_test": y_test_orig, "X_train": X_train_orig, "y_train": y_train_orig, "GP": GP_pred, "error": sigma, "linear": lin_pred}
    fname = args.parameter+'_pickle'
    if args.run_tag:
        fname = args.run_tag + '_' + fname
    pickle.dump(data, open(fname, "wb"))
