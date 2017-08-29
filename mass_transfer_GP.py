### INITIALIZATION SECTION ###

# import packages
import numpy as np
import scipy as sp
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import norm
import argparse
import time
import os
import pdb
import multiprocessing
import itertools
import pickle
import mkl

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn import gaussian_process
GP = gaussian_process.GaussianProcess # deprecated
GPR = gaussian_process.GaussianProcessRegressor
GPR_prior = gaussian_process.GaussianProcessRegressor_prior
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern as matern, ExpSineSquared as ess
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
print "\nScikit-learn version: %s" % (sklearn.__version__)


# read in arguments
argp = argparse.ArgumentParser()
argp.add_argument("-g", "--grid-path", type=str, default='fine_grid', help="Grid of MT sequences you wish to use. Default='fine_grid'.")
argp.add_argument("-r", "--resamp", type=str, default='all_l2_10', help="Resampled tracks you wish to use for interpolation. Default='all_l2_10'.")
argp.add_argument("-p", "--parameter", type=str, help="Parameter you wish to interpolate. Parameters of interest include: star_1_mass, star_2_mass, log_Teff, log_L, log_R, period_days, age, log_dt, log_abs_mdot, binary_separation, lg_mtransfer_rate, lg_mstar_dot_1, lg_mstar_dot_2, etc. Note: star_1 = companion, star_2 = black hole.")
argp.add_argument("-t", "--test-set", type=float, default=0.2, help="Fraction of the total (or cut set if cut-set is specified) set that is held out for testing (i.e., the GP will be trained on the N*(1-t) datapoints). Default = 0.2.")
argp.add_argument("-c", "--cut-set", type=float, default=None, help="Randomly reduces the number of tracks by N*(c) so the input matrix isn't too crazy big. Default=None.")
argp.add_argument("-rs", "--random-seed", type=int, help="Use this for reproducible output.")
argp.add_argument("-nc", "--num-cores", type=int, default=None, help="Specify number of cores to use. If not specifed, will parallelize over all available cores. If 1 core is specified, will circumvent multiprocessing for debugging purposes.")
argp.add_argument("-f", "--run-tag", help="Use this as the stem for all file output.")
argp.add_argument("-pc", "--principal-component", type=int, default=None, help="Use this option to specify that you would like to interpolate the prinripal components rather than the steps, and specifed the number of PCs you would like to retain. Default=None.")
argp.add_argument("-s", "--save-pickle", action="store_true", help="Save the GP model as a pickle. Default is off.")
argp.add_argument("-pl", "--make-plots", action="store_true", help="Makes and saves plots during this script. Default is off.")
argp.add_argument("-e", "--expand-matrix", action="store_true", help="Use this option to expand the input/output matrices, so that steps/principal components retain some correlation information. This will expand the matrices by the length of the resampling, or by the number specified in PC if args.principal_component is defined. Note that this will increase training time by ~x^3 and may cause memory issues. Default=None.")
argp.add_argument("-T", "--test-MT", type=int, default=None, help="Cuts the dataset to make a smaller grid centered around a testing track. Providing an integer gives the number nearest tracks you wish to keep in the training set. Default='None'. Default testing track using this option is 10 Mbh, 15 M2, 2 P, 0.02 Z. Default path for test_MT resamplings is 'data/test_MT/resampled/<resamp>/'.")
argp.add_argument("-pr", "--prior", type=str, default=None, help="This will call a different Gaussian Process Regressor that imposes a prior on the hyperparameters, used when there is intuition for what the scale length/amplitude in the kernel should approximately be. One can add additional priors in the GaussianProcessRegressor_prior function in gpr.py, which is located in the scikit library. Default=None.")
args = argp.parse_args()




### SETUP SECTION ###

# if we are using PCs, use a 1000-step resampling
if args.principal_component:
    args.resamp = 'all_l2_1000'

# path to the directory that has all the resampled files you wish to use
basepath = os.path.dirname(os.path.realpath(__file__))
path = basepath + '/data/' + args.grid_path + '/resampled/' + args.resamp + '/'


# dict of parameter names
param_names=['log_dt','log_abs_mdot','he_core_mass','c_core_mass','o_core_mass','mass_conv_core','log_LH','log_LHe','log_LZ','log_Lnuc','log_Teff','log_L','log_R','log_g','surf_avg_omega','surf_avg_omega_div_omega_crit','center_h1','center_he4','center_c12','center_o16','surface_c12','surface_o16','total_mass_h1','total_mass_he4','log_center_P','log_center_Rho','log_center_T','model_number','age','period_days','binary_separation','v_orb_1','v_orb_2','star_1_radius','rl_1','rl_2','rl_relative_overflow_1','rl_relative_overflow_2','star_1_mass','star_2_mass','lg_mtransfer_rate','lg_mstar_dot_1','lg_mstar_dot_2','lg_system_mdot_1','lg_system_mdot_2','lg_wind_mdot_1','lg_wind_mdot_2','xfer_fraction','J_orb','Jdot','jdot_mb','jdot_gr','jdot_ml','jdot_ls','jdot_missing_wind','extra_jdot','donor_index','point_mass_index','r_tau100','r_tau1000','t_dynamical_tau100','t_dynamical_tau1000','m_dynamical_tau100','m_dynamical_tau100_1','t_thermal_tau1000','t_thermal_tau1000_1','m_thermal_tau100','m_thermal_tau1000']

# find the index of the parameter of interest
param_idx = param_names.index(args.parameter)
print '\nParameter(s) for interpolation: %s' % args.parameter


# read in inputs and outputs
inputs=[]
outputs=[]
for file in os.listdir(path):
    x = np.load(path + file)
    inputs.append((np.float(x['M2_init']),np.float(x['Mbh_init']),np.log10(np.float(x['P_init'])),np.log10(np.float(x['Z_init']))))
    outputs.append(x['x_resamp'][:,param_idx])


# reshape as arrays
inputs = np.reshape(inputs,[len(inputs),4]) # 4 inputs
outputs = np.asarray(outputs)
resamp_len = outputs.shape[1]
print '\nThe full grid contains %i tracks' % len(inputs)

# store inputs as a dataframe
inputs_df = pd.DataFrame({"M2_init": inputs[:,0], "Mbh_init": inputs[:,1], "P_init": inputs[:,2], "Z_init": inputs[:,3]})
full_inputs_df = inputs_df.copy()   # store the full input grid for renormalization purposes

# if cut_set is specified, we randomly reduce the set to args.cut_set of the total
if args.cut_set:
    reduced = np.where(np.random.random(size=len(inputs)) < args.cut_set)
    inputs = inputs[reduced]
    outputs = outputs[reduced]
    inputs_df = pd.DataFrame({"M2_init": inputs[:,0], "Mbh_init": inputs[:,1], "P_init": inputs[:,2], "Z_init": inputs[:,3]})
    print 'The cut grid contains %i tracks' % len(inputs)
    


print '\nThe bounds of the grid for interpolation are:'
print '   M2_init: %f - %f' % (inputs_df["M2_init"].min(), inputs_df["M2_init"].max())
print '   Mbh_init: %f - %f' % (inputs_df["Mbh_init"].min(), inputs_df["Mbh_init"].max())
print '   P_init: %f - %f' % (10**inputs_df["P_init"].min(), 10**inputs_df["P_init"].max())
print '   Z_init: %f - %f' % (10**(inputs_df["Z_init"]).min(), 10**(inputs_df["Z_init"]).max())


# plot the parameter space coverage
if args.make_plots:
    fig=plt.figure(figsize = (12,8), facecolor = 'white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel('$Black\ Hole\ Mass\ (M_{\odot})$', rotation=0, labelpad=20, size=12)
    ax.set_ylabel('$Companion\ Mass\ (M_{\odot})$', rotation=0, labelpad=20, size=12)
    ax.set_xlabel('$Log\ Period\ (s)$', rotation=0, labelpad=20, size=12)

    pts = ax.scatter(inputs_df['P_init'], inputs_df['M2_init'], inputs_df['Mbh_init'], zdir='z', s=5, cmap='viridis', c=inputs_df['Z_init'], label='simulated tracks')
    fig.colorbar(pts)
    plt.tight_layout()
    plt.legend()
    fname = 'init_condit.png'
    if args.run_tag:
        fname = args.run_tag + '_' + fname
    plt.savefig(fname)



# define function to normalize inputs to be centered at 0 with a std of 1 #FIXME this is just normalizing inputs to be between [0,1]
def normalize(df, norm_df):
    normed = df.copy()
    for key in df:
        normed[key] = (df[key]-norm_df[key].min()) / (norm_df[key].max()-norm_df[key].min())
    return normed
def denormalize(df, norm_df):
    denormed = df.copy()
    for key in df:
        denormed[key] = (df[key] * (norm_df[key].max()-norm_df[key].min()) + norm_df[key].min())
    return denormed

# define function to "center" the output values by subtracting out the mean value
def center(y):
    means=[]
    y_cen = np.empty(np.shape(y))
    for i in xrange(len(y.T)):   # need to iterate over all the steps
        means.append(y[:,i].mean())
        y_cen[:,i] = y[:,i]-y[:,i].mean()
    return y_cen, means

def uncenter(y_cen, means):
    y = np.empty(np.shape(y_cen))
    for i in xrange(len(y_cen.T)):
        y[:,i] = y_cen[:,i]+means[i]
    return y

# normalize & center
inputs = normalize(inputs_df, full_inputs_df)
outputs, means = center(outputs)


# split dataset into training and testing sets
if args.random_seed:
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(inputs, outputs, test_size=args.test_set, random_state=args.random_seed)
else:
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(inputs, outputs, test_size=args.test_set)



## If test_MT is specified, read in this track and cut the inputs tracks as specified #FIXME this won't work for >1 test_MT
if args.test_MT:
    testMT_path = basepath + '/data/test_MT/resampled/' + args.resamp + '/'
    testMT_inputs=[]
    testMT_outputs=[]
    for file in os.listdir(testMT_path):   # should only be 1 file in this directory, but can have more
        x = np.load(testMT_path + file)
        testMT_inputs.append((np.float(x['M2_init']),np.float(x['Mbh_init']),np.log10(np.float(x['P_init'])),np.log10(0.02)))   # for some reason no Z recorded...
        testMT_outputs.append(x['x_resamp'][:,param_idx])

    # reshape as arrays
    testMT_inputs = np.reshape(testMT_inputs,[len(testMT_inputs),4])
    testMT_outputs = np.asarray(testMT_outputs)

    # store test_MT inputs as a dataframe
    testMT_inputs_df = pd.DataFrame({"M2_init": testMT_inputs[:,0], "Mbh_init": testMT_inputs[:,1], "P_init": testMT_inputs[:,2], "Z_init": testMT_inputs[:,3]})

    # normalize test_MT inputs & center from testMT_outputs
    testMT_inputs = normalize(testMT_inputs_df, full_inputs_df)
    testMT_outputs = testMT_outputs - means

    # choose args.test_MT tracks closest to the testing track as the training data
    norm_vecs=[]
    for inp in np.array(inputs):
        norm_vecs.append(np.linalg.norm(np.array(testMT_inputs)-inp))
        sorted_idxs = np.argsort(norm_vecs)

    # only use the closest input and output points
    inputs_df = inputs_df.iloc[sorted_idxs[0:args.test_MT]]
    outputs = outputs[sorted_idxs[0:args.test_MT]]
    print '\nThe cut grid surrounding the testing track contains:'
    print '   M2_init: %f - %f' % (inputs_df["M2_init"].min(), inputs_df["M2_init"].max())
    print '   Mbh_init: %f - %f' % (inputs_df["Mbh_init"].min(), inputs_df["Mbh_init"].max())
    print '   P_init: %f - %f' % (10**inputs_df["P_init"].min(), 10**inputs_df["P_init"].max())
    print '   Z_init: %f - %f' % (10**inputs_df["Z_init"].min(), 10**inputs_df["Z_init"].max())

    # re-normalize the cut array so that its values range from 0-1
    testMT_inputs = normalize(testMT_inputs_df, inputs_df)
    inputs = normalize(inputs_df, inputs_df)

    # split dataset into training and testing sets
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = inputs, testMT_inputs, outputs, testMT_outputs



# add an extra dimension of 'step' to the normalized inputs and flatten if args.expand_matrix is specified
if args.expand_matrix:
    step_space = np.linspace(0,resamp_len,resamp_len)/resamp_len
    X_train = np.array(list(np.append(x,y) for x in np.array(X_train_orig) for y in step_space))
    X_test = np.array(list(np.append(x,y) for x in np.array(X_test_orig) for y in step_space))
    y_train = y_train_orig.flatten()
    y_test = y_test_orig.flatten()
else:
    X_train = X_train_orig; X_test = X_test_orig
    y_train = y_train_orig; y_test = y_test_orig



### PRINCIPAL COMPONENET SECTION ###

# convert y data using PCs if args.principal-components is specified
if args.principal_component:
    print "\nInterpolating using prinicpal components"
    print "   n_components: %i" % args.principal_component
    pca = PCA(n_components=args.principal_component)
    pca.fit(y_train_orig)
    if args.expand_matrix:
        step_space = np.linspace(0,args.principal_component,args.principal_component)/args.principal_component
        X_train = np.array(list(np.append(x,y) for x in np.array(X_train_orig) for y in step_space))
        X_test = np.array(list(np.append(x,y) for x in np.array(X_test_orig) for y in step_space))
        y_train = pca.transform(y_train_orig)
        y_train = y_train.flatten()
        y_test = pca.transform(y_test_orig)
        y_test = y_test.flatten()
    else: 
        X_train = X_train_orig; X_test = X_test_orig
        y_train = pca.transform(y_train_orig)
        y_test = pca.transform(y_test_orig)   # hold onto these in case we want to look at them
        

### INTERPOLATION SECTION ###

# define interpolation functions
# notation for parallelizing: data[0]: X_train, data[1]: y_train, data[2]: X_test

def GPR_scikit(data):
    X = np.atleast_2d(data[0])
    y = np.atleast_2d(data[1]).T # need to transpose to make dimensions match since y is 1D

    # NOTE: kernel uses log of given hyperparameters, such that theta=np.log(length_scale)
    cs=[1e-1]
    csb=[(1e-3,1e0)]
    ls=[1e-2]*np.shape(inputs_df)[1]
    lsb=[(1e-5,1e-0)]*np.shape(inputs_df)[1]
    kernel = C(cs,csb) * RBF(length_scale=ls, length_scale_bounds=lsb)
    # specify the parameters for the prior (in the case of log_normal, mean & scale)
    prior_params = [(-1,0.5)]+[(-2,0.5)]*np.shape(inputs_df)[1]

    yerr = 1e-6

    # choose between regular and prior-included GP
    if args.prior:
        gp = GPR_prior(kernel=kernel, n_restarts_optimizer=9, normalize_y=False, prior=args.prior, prior_params=prior_params)
        gp.fit_prior(X, y)
    else: 
        gp = GPR(kernel=kernel, n_restarts_optimizer=9, normalize_y=False)
        gp.fit(X, y)

    params = gp.kernel_.get_params()

    X_pred = np.atleast_2d(data[2])
    y_pred, sigma = gp.predict(X_pred, return_std=True)
    y_pred = np.reshape(y_pred,len(y_pred))

    return y_pred, sigma, params


def linear_interp(data):
    value = sp.interpolate.griddata(data[0],data[1],data[2], method='linear', fill_value=0.0)
    return value


# specify number of cores to run on
if args.num_cores:
    num_cores = args.num_cores
else:
    num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)


# do the GP inteprolation
start_time = time.time()   # start the clock

if y_train.ndim == 1:
    y_train = np.expand_dims(y_train,1)   # adds dimension if array is 1-dimensional
temp=[]
for item in y_train.T:
    combined = (X_train, item, X_test)
    temp.append(combined) # now, temp is holding all the information needed for interpolation function

if args.prior:
    print '\nUsing %s prior on the likelihood calculations of hyperparameters' % args.prior

if args.num_cores==1:
    results=[]
    for i in temp:
        results.append(GPR_scikit(i))
else:
    results = pool.map(GPR_scikit, temp) # this is the workhorse line

GP_pred = list(results[i][0] for i in xrange(len(results)))
sigma = list(results[i][1] for i in xrange(len(results)))
params = list(results[i][2] for i in xrange(len(results)))

elapsed = time.time() - start_time   # See how long it took
print '\nDone with GP interpolation for %s...it only took %f seconds!' % (args.parameter,elapsed)



# do the linear inteprolation
start_time = time.time()   # start the clock

if y_train.ndim == 1:
    y_train = np.expand_dims(y_train,1)   # adds dimension if array is 1-dimensional
temp=[]
for item in y_train.T:
    combined = (X_train, item, X_test)
    temp.append(combined) # now, temp is holding all the information needed for interpolation function

if args.num_cores==1:
    results=[]
    for i in temp:
        results.append(linear_interp(i))
else:
    results = pool.map(linear_interp, temp)

lin_pred = list(results[i] for i in xrange(len(results)))

elapsed = time.time() - start_time   # See how long it took
print 'Done with linear interpolation for %s...it only took %f seconds!\n' % (args.parameter,elapsed)



### STRUCTURE OUTPUT AND SAVE ###

# reshape outputs to be in form (tracks, steps/PCs)
if args.principal_component:
    GP_pred_PC = np.reshape(np.transpose(GP_pred), (len(y_test_orig),args.principal_component))
    sigma_PC = np.reshape(np.transpose(sigma), (len(y_test_orig),args.principal_component))
    lin_pred_PC = np.reshape(np.transpose(lin_pred), (len(y_test_orig),args.principal_component))
else:
    GP_pred = np.reshape(np.transpose(GP_pred), (y_test_orig.shape[0],y_test_orig.shape[1]))
    sigma = np.reshape(np.transpose(sigma), (y_test_orig.shape[0],y_test_orig.shape[1]))
    lin_pred = np.reshape(np.transpose(lin_pred), (y_test_orig.shape[0],y_test_orig.shape[1]))

# if PC was specified, return data to original basis
if args.principal_component:
    GP_pred = pca.inverse_transform(GP_pred_PC)
    sigma = pca.inverse_transform(sigma_PC)   # FIXME this probably isn't giving the correct error
    lin_pred = pca.inverse_transform(lin_pred_PC)


# denormalize the input testing and training sets
X_test_orig = denormalize(X_test_orig, inputs_df)
X_train_orig = denormalize(X_train_orig, inputs_df)

# uncenter output values
GP_pred = uncenter(GP_pred, means)
lin_pred = uncenter(lin_pred, means)
y_test_orig = uncenter(y_test_orig, means)
y_train_orig = uncenter(y_train_orig, means)

print params


# save pickle
if args.save_pickle:
    data = {"inputs": inputs_df, "full_inputs": full_inputs_df, "X_test": X_test_orig, "y_test": y_test_orig, "X_train": X_train_orig, "y_train": y_train_orig, "GP": GP_pred, "error": sigma, "linear": lin_pred, "params": params}
    if args.principal_component:
        data["GP_PC"] = GP_pred_PC
        data["error_PC"] = sigma_PC
        data["lin_PC"] = lin_pred_PC
        data["y_test_PC"] = y_test
    fname = args.parameter+'_pickle'
    if args.run_tag:
        fname = args.run_tag + '_' + fname
    if not os.path.exists("pickles/"):
        os.makedirs("pickles/")
    pickle.dump(data, open("pickles/" + fname, "wb"))

