import numpy as np
from scipy import interpolate
import glob
import sys
import astropy.io.ascii
import os

#------------------------
# User input!
# The list_x can have any number of column names found in the npy files.
list_x = ['star_1_mass' , 'log_Teff', 'log_L', 'log_R', 'period_days', 'lg_mstar_dot_1', 'age']
norm = 2.       # FLOAT: choose between L^1, L^2, ... L^N normalization
Nresample = 1000
#------------------------

list_x.sort() # put in alphabetical order to prevent multiple of the same resamplings being saved
#dir_path = '-'.join(list_x) + '-' + str(norm) + '/'
dir_path = 'all' + '_l' + str(norm)[0] + '_' + str(Nresample) + '/'

# Path to data
path_in = '/projects/b1011/mzevin/gaussian_process_MT/data/test_MT/npy/'
# Path to log file
path_log = '/projects/b1011/mzevin/gaussian_process_MT/data/log/'
# Path to write out new data
sys_path = '/projects/b1011/mzevin/gaussian_process_MT/data/test_MT/resampled/'

# Check to see if the resampling already exists in path_out
path_out = sys_path + dir_path
if os.path.exists(path_out):
    print 'This resampling already exists'
    sys.exit(0)
else:
    os.makedirs(path_out)

# Define functions used for resampling
def norm_curve(p):
    return (p - np.min(p))/(np.max(p)-np.min(p))

def euclidean_path(x,list_x,n_start,n_end,norm):
    # list_x: list of history column names from MESA
    y = np.zeros(n_end-n_start+1)
    for name in list_x:
        n = norm_curve(x[name][n_start:n_end+1])
        y[1:] = y[1:] + np.abs((n[0:-1]-n[1:]))**norm
    f = (y)**(1/norm)
    return np.cumsum(f)

# Read log file
log = astropy.io.ascii.read(path_log+'20160606complete_log.csv')

# Read files for resampling:
files_to_resample = glob.glob(path_in+"*.npy")
Nfiles = np.size(files_to_resample)


counter = 0         # Loop in this way to keep track of progress
for i in range(counter,Nfiles):
    fname = files_to_resample[i]

    # Get name for saving
    dum=fname.split('/')
    name_out = dum[-1][:-4]

    # Print progress
    counter = counter +1
    sys.stdout.write("Progress: %d%%   \r" % (float(counter)/float(Nfiles) *100.) )
    sys.stdout.flush()

    # Load file
    x=np.load(fname)
    FieldNames = x.dtype.names
    Noriginal = len(x[FieldNames[0]]) # In case we need it


    # Make sure that there are more than 3 datapoints, otherwise start loop over
    if len(x['age']) <= 3.:
        continue

    # Check that the age is increasing in lines of history.data
    # Otherwise, MESA may have done a back-up in which case we should remove these profiles
    idx_lines_to_keep = np.where(x['age'][:-1] < x['age'][1:])[0]
    x = x[:][idx_lines_to_keep]

    # Only choose the part of the track where mass-trasfer is ongoing
    # In case of two mass-transfer episodes separated by a detached phase the detached phase is included
    # However, the quality of the resampling in this case has to be checked
    idx_mdot = np.where(x['log_abs_mdot'] > -15)

    # Specify start and end of mass transfer sequence
    n_start = idx_mdot[0][0]
    n_end = idx_mdot[0][-1]

    # Output log file
    logout = log[np.where(log['name'] == name_out)]
    # Output initial conditions
    Z_init = logout['Z']
    M2_init = x['star_1_mass'][n_start]
    Mbh_init = x['star_2_mass'][n_start]
    P_init = x['period_days'][n_start]
    # Output matrix with axes of [timesteps, fields]
    x_resamp = np.zeros((Nresample,  len(FieldNames)))

    # Check if the system undervent mass transfer.
    if np.abs(x['star_1_mass'][-1] - M2_init) < 1e-5:
        continue


    # Interpolate in terms of the normalized distance s traversed by a set of parameters
    x_int_low = 0.0000001
    x_int_high = 0.9999999
    x_int = np.linspace(x_int_low, x_int_high, Nresample)

    # Normalised individual parameter
    s = euclidean_path(x,list_x,n_start,n_end,norm)
    xp_int = norm_curve(s)
    for j in xrange(len(FieldNames)):
        fp_int = x[FieldNames[j]][n_start:n_end+1]
        f = interpolate.interp1d(xp_int,fp_int)
	x_resamp[:,j] = f(x_int)

    # Save output as npz file
    fout = fname.split('.')
    fout = fout[0]
    np.savez(path_out+name_out+".npz", M2_init=M2_init, Mbh_init=Mbh_init, P_init=P_init, Z_init = Z_init, x_resamp = x_resamp, log=logout)
