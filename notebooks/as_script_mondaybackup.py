# %%

# data from https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/
from cmath import isinf
import os
from click import getchar
from tqdm.notebook import tqdm

import math
import threading

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns

from py_files.extract import extract_data, extract_metadata
from bycycle import features, cyclepoints, plts
from scipy.signal import find_peaks
from scipy.signal import resample
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.fft import fft
from scipy.stats import pearsonr
from neurodsp.filt import filter_signal
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.spectral import compute_spectrum

import statsmodels.api as sm

# Import filter function
from neurodsp.filt import filter_signal

import bycycle

from matplotlib.pyplot import cm

from IPython.display import display

# %matplotlib inline

# 25, 5, 10, 15
# 10 currently works best, sample size: 1
# data 0: 10
H_VAL = 10
DATA_ID = 133
MULTIPLE_DATA_IDS = [33,133,233,333,433]
# was 1
FREQ_HIGHPASS = 1

default_a = [2000, -2000, 2000, -2000, 2000]
default_b = [100, 200, 300, 400, 500]
default_c = [-0.001, -0.001, -0.001, -0.001, -0.001]
default_d = [0, 0, 0, 0, 0]

default_guess_a ,default_guess_c,default_guess_d = 2000,-0.001,0
param_sums = np.zeros(4)
param_count = 0

# beginning at index 0 and increasing, find_max returns an index and value of the max value in the time series
def find_max(time_series=None, left=np.inf, right=-np.inf):
    errouneous_output = (-1, -np.inf)
    # bound checking
    if len(time_series) == 0:
        return errouneous_output
    if left < 0 or left > len(time_series):
        return errouneous_output
    if left > right:
        return errouneous_output
    if right > len(time_series):
        return errouneous_output
    # corner case
    if left == right:
        # super trivial result but what else?
        return (left, time_series[left])

    idx = -1
    max_val = time_series[0]
    for i in range(left, right):
        if time_series[i] > max_val:
            idx = i
            max_val = time_series[i]
    return(idx, max_val)


# find_max for min value
def find_min(time_series=None, left=np.inf, right=-np.inf):
    errouneous_output = (-1, -np.inf)
    # bound checking
    if len(time_series) == 0:
        return errouneous_output
    if left < 0 or left > len(time_series):
        return errouneous_output
    if left > right:
        return errouneous_output
    if right > len(time_series):
        return errouneous_output
    # corner case
    if left == right:
        # super trivial result but what else?
        return (left, time_series[left])
    idx = -1
    max_val = time_series[0]
    for i in range(left, right):
        if time_series[i] < max_val:
            idx = i
            max_val = time_series[i]
    return(idx, max_val)


# (f(x+h)-f(x))/h, where h is an integer
def disc_derivative(time_series, x_val, h_val):
    if x_val+h_val in range(0, len(time_series)):
        deriv = (time_series[x_val+h_val]-time_series[x_val])
        if deriv == 0:
            return 0
        if deriv > 0:
            return 1
        else:
            return -1
    return int(0)

# finds points where polarity of discrete derivative changes
def time_series_polarity_change(time_series, idx, offset, direction):
    if direction < 0:
        offset = -offset
    extreme_bound = None
    if direction > 0:
        extreme_bound = len(time_series)
    else:
        extreme_bound = 0
    initial_derivative = disc_derivative(
        time_series=time_series, x_val=idx, h_val=offset)
    while (initial_derivative == 0) and (idx in range(len(time_series))):
        idx += offset
        initial_derivative = disc_derivative(
            time_series=time_series, x_val=idx, h_val=offset)
    for i in range(idx, extreme_bound, offset):
        new_derivative = disc_derivative(
            time_series=time_series, x_val=i, h_val=offset)
        if new_derivative != initial_derivative:
            return i

    return -1

#  this function plots lines between points on the time series starting
# at index i=0 and increasing by multiples of interval,
def linear_smoothing(time_series, interval):
    if type(interval) != int or len(time_series) == 0 or interval <= 0:
        return None
    retArray = np.zeros(len(time_series))
    iterations = math.floor(float(len(time_series))/float(interval))
    remainder = len(time_series) % interval
    for i in range(iterations):
        y2 = time_series[((i+1)*interval)-1]
        y1 = time_series[i*interval]
        # print(interval)
        slope = float(y2 - y1)/float(interval)
        for j in range(interval):
            retArray[i*interval + j] = time_series[i*interval] + j*slope
    if remainder > 0:
        # print("remainder "+str(remainder))
        i=iterations*interval
        y2 = time_series[i+remainder-1]
        y1 = time_series[i]
        slope = float(y2 - y1)/float(interval)
        for i in range(remainder):
            retArray[iterations*interval +
                     i] = time_series[iterations*interval] + i*slope
    return retArray


# returns time series for an unskewed gaussian
# with time series x, amplitude a, center index b, and 'negative narrowness' c.
def gaussian(x, a, b, c):
    return a*np.exp(c*((x-b)**2))


# returns time series for a skewed gaussian
# with time series x, amplitude a, center index b, and 'negative narrowness' c,
# and skew parameter d
def skewed_gaussian(x, a, b, c, d):
    center = b
    skew_param = d
    normal_gaussian = gaussian(x, a, b, c)
    cdf = norm.cdf(skew_param*(x-center))
    skewed_function = 2*cdf*normal_gaussian
    return skewed_function

# reduces an n-dimensional 4-tuple to an n-1-dimensional 4*n-tuple
# by unpacking each element of each element of the 4-tuple in the
# order: a0,b0,...,d0,a1,...,d(n-2),a(n-1),...,d(n-1)
def flatten_guesses(a, b, c, d):
    if len(a) != len(b) or len(c) != len(d):
        return "bad shape"
    if len(a) != len(c):
        return "bad shape"

    guesses = []
    lists = [a, b, c, d]
    for i in range(len(a)):
        for j in range(len(lists)):
            guesses.append(lists[j][i])        
    # print(len(guesses))
    return guesses

# inverse to flatten_guesses
def build_guesses(width, *params):
    retVal=[]
    if len(params)%width!=0:
        print("length "+str(len(params)) + " is not divisible by "+ str(width))
        return None
    for i in range(width):
        retVal.append([])
    for i in range (len(params)):
        retVal[i%width].append(params[i])
    return retVal

# plots n superpositioned gaussian where n is the length of a,b,c, and d.
# a,b,c, and d are the (amp, center, neg narrowness, skew) values, respectively, for skewed_gaussian()
def n_skewed_gaussian(x, a, b, c, d):
    length = len(a)
    if len(b) != length or len(c) != length or len(d) != length:
        return "invalid arg set. a,b,c,d not all equal length"
    retData = np.zeros(len(x))
    for i in range(length):
        retData = retData+skewed_gaussian(x, a[i], b[i], c[i], d[i])

    return retData

# n_skewed_gaussian with (a,b,c,d) expanded
def CURVE_FIT_5_skewed_gaussian(x,*params):
    newParams=build_guesses(4,*params)
    return n_skewed_gaussian(x,*newParams)
    
# recursively prints shapes of lists. remove
def print_list_shape(list):
    shadow=list
    try:
        while len(shadow) > 0:
            # print(len(shadow))
            shadow=shadow[0]
    except:
        return
    return

# plots unskewed gaussian
def plot_gaussian(x, a, b, c):
    func = a*np.exp(c*((x-b)**2))
    x = np.linspace(0, len(func), func)
    plt.plot(x, func)


# fits single gaussian to single signal. deprecated a bit?
def fit_curve(sig=None, p0=[], bounds=()):
    p, _ = curve_fit(f=skewed_gaussian, xdata=np.linspace(
        0, len(sig), len(sig)), ydata=sig, p0=p0, bounds=bounds)
    peakiness = gaussian_peakiness(sig, 0, len(sig), p[0])
    np.append(p, peakiness)
    return p

# this function should fit the curve as accurately as possible to n gaussians



# use all bounds
def fit_n_gaussian_general(x=None, n=0):

    # this function should fit the curve as accurately as possible with b values bounded by a list of ordered pairs.
    # designed for convenience.
    return -1

# generate flat bounds from tuple bounds
def generate_bounds(a, b, c, d):
    errstring = "wrong shape"
    errstring_1 = "empty argument"
    list = [a, b, c, d]
    retBounds = ([], [])
    
    
    if len(a) != len(b) or len(c) != len(d):
        return errstring
    if len(b) != len(d):
        return errstring
    if len(a) == 0:
        return errstring_1
    
    # shape check
    for i in range(len(list)):
        bound_set = list[i]
        for i in range(len(bound_set)):
            bound = bound_set[i]
            if len(bound) != 2:
                return errstring


    bound_length = len(a)
    for i in range(bound_length):
        for j in range(len(list)):
            bound_pair = list[j]
            currBound = bound_pair[i]
            retBounds[0].append(currBound[0])
            retBounds[1].append(currBound[1])
    return retBounds


# fits n gaussian curves, with bounds on center points specified.
# works well??
def fit_n_gaussian_center_bounds(x, b_bounds):
    global param_count
    global param_sums
    if len(b_bounds[0]) != len(b_bounds[1]):
        return "bounds have different lengths"

    sz = len(b_bounds)
    a, c, d = [], [], []
    b = []
    a_guesses = []
    b_guesses = []
    c_guesses = []
    d_guesses = []
    guesses=None
    
    for i in range(sz):
        a.append((-1e20, 1e20))
        c.append((-1, 0))
        d.append((-1e20, 1e20))
    for i in range(sz):
        curr_bound = b_bounds[i]
        if np.isinf(curr_bound[0]) or np.isinf(curr_bound[1]):
            return "no infinite bounds supported yet. Please, use finite bounds."
        left = curr_bound[0]
        right = curr_bound[1]
        if left >= right:
            return "left bound >= right bound"
        b_guesses.append((left+right)/2)
        b.append((left, right))
        
        a_guesses.append(default_guess_a)
        c_guesses.append(default_guess_c)
        d_guesses.append(default_guess_d)

    # we must handle all good infinity edge cases, out of bounds, and the likes..

    bounds = generate_bounds(a, b, c, d)
    # print(len(a_guesses))
    # print(len(b_guesses))
    # print(len(c_guesses))
    # print(len(d_guesses))

    if param_count==0:
        guesses=flatten_guesses(a_guesses,b_guesses,c_guesses,d_guesses)
    else:
        guesses = np.zeros(4)
        for i in range(len(param_sums)):
            guesses[i]=param_sums[i]/param_count
    print_list_shape(guesses)
    for i in range(len(bounds[0])):
        if bounds[0][i] >= bounds[1][i]:
            print("bound at index "+ str(i)+" violates le/gt relationship")
    # print("bounds: "+str(bounds))
    # print("guesses: "+str(guesses))

    for i in range(len(x)):
        element = x[i]
        if np.isinf(element) or np.isnan(element):
            x[i]=0
    
    # TODO: figure out why df doesn't have 4 gaussians per entry
    params, _ = curve_fit(f=CURVE_FIT_5_skewed_gaussian, xdata=np.linspace(0, len(x), len(x)),
                    ydata=x,
                    p0=guesses,
                    #    bounds=bounds,
                    # we usually use lm
                    method='lm',
                    maxfev=100000)
    # print("params: "+str(params))
    # X0 must be one-dimensional...let's flatten the array.
    for i in range(len(params)):
        param_sums[i%4]+=params[i]
    param_count+=1
    return params


# tgt is y value, time_series is the signal, left,right are bounds
# this function sweeps from 0 to len(time_series)
def find_intercepts_in_range(time_series, tgt, left, right):
    leq = bool(time_series[left] <= tgt)
    retslice = []
    for i in range(left, right):
        if leq and time_series[i] > tgt:
            retslice.append(i)
            leq = False
        elif (not leq) and time_series[i] <= tgt:
            retslice.append(i)
            leq = True
    return retslice

# this function begins at center, and finds the nearest left and right intercepts for the tgt y-value
# BUG: returned "[] and 123"
def find_intercepts_telescoping(time_series, tgt, left_b, right_b, center):
    in_exit = False
    # should be fine
    # if center==0:
    #     in_exit=True
    print("tgt", tgt, "left_b:", left_b, "right_b:", right_b, "center", center)
    left = []
    right = []

    left_b = int(left_b)
    right_b = int(right_b)
    # invalid arguments
    if left_b < 0 or right_b < 0 or left_b > right_b:
        return (left, right)
    if right_b > len(time_series):
        right_b = len(time_series)
    for i in range(center-left_b-1):
        last = time_series[center-i]
        this = time_series[center-i-1]
        # print("first loop: last="+str(last)+", this="+str(this))
        if last > tgt and this <= tgt:
            left.append(int(center-i-1))
            break
        if last <= tgt and this > tgt:
            left.append(int(center-i-1))
            break
    for i in range(right_b-center-1):
        last = time_series[center+i]
        this = time_series[center+i+1]
        # print("second loop: last="+str(last)+", this="+str(this))
        if last > tgt and this <= tgt:
            right.append(int(center+i+1))
            break
        if last <= tgt and this > tgt:
            right.append(int(center+i+1))
            break

    print("should be returning: "+str(left)+" and "+str(right))
    # should be fine
    # if(left == []):
    #     print(in_exit)
    #     os._exit(1)
    return left, right

# ideally, this function comments about the peakiness of an individual gaussian. I am tired, tho
def gaussian_peakiness(time_series, left, right, pol):
    H_VAL_PEAKINESS = 1

    if pol >= 0:
        max_idx, _ = find_max(time_series=time_series, left=left, right=right)
    else:
        max_idx, _ = find_min(time_series=time_series, left=left, right=right)

    if max_idx == -1:
        return 0
    peakedness = abs(time_series[max_idx] - time_series[max_idx-H_VAL_PEAKINESS]) + abs(
        time_series[max_idx] - time_series[max_idx+H_VAL_PEAKINESS])
    return peakedness

def shorten(sig, init_freq, tgt_freq):
    old_len = len(sig)
    new_len = int(float(tgt_freq)/float(init_freq)*float(len(sig)))
    result = resample(sig, new_len)
    return result, int((float(new_len)/float(old_len))*float(init_freq))

def scale(val, orig, new):
    return float(float(val)*float(orig)/float(new))

# %%
x=[0,1,2,3]
x=np.append(x,4)

x

# %%
# Data paths
# I've included the whole path because uploading all the data isn't ideal for me.
dir_path = '/Users/kenton/HOME/coding/python/ecg_param-kenton/data'
files_dat = [i for i in sorted(os.listdir(dir_path)) if i.endswith('dat')]
files_hea = [i for i in sorted(os.listdir(dir_path)) if i.endswith('hea')]
files_hea = [i for i in files_hea if i != '0400.hea'] # missing one participant's data
# for DATA_ID in MULTIPLE_DATA_IDS:
for DATA_ID in range(0,50):
    print(DATA_ID)
    # Extract single subject
    sigs, metadata = extract_data(
        os.path.join(dir_path, files_dat[DATA_ID]),
        os.path.join(dir_path, files_hea[DATA_ID]),
        raw_dtype='int16'
    )



    # Each subject has three 'channels': two ecg and one pulse
    # not really
    # for i in range(len(sigs)):
    #     plt.figure(i, figsize=(16, 4))
    #     plt.plot(sigs[i][500:1500])
    #     plt.title('FIG '+str(i), size=20)

    # %% [markdown]
    # ## Finding peaks with Scipy

    # %%
    fs=1000
    
    data = sigs[0]
    print(type(data))
    # data, fs = np.array(shorten(data,1000,120))
    print(fs)
    print(data.shape)
    #fs*num=1000
    f_range_hi = (FREQ_HIGHPASS, None)

    # %%
    # Highpass filter to remove drift
    sig_filt = filter_signal(data, fs, 'highpass', f_range_hi)
    # data, fs = np.array(shorten(data,1000,500))
    sig_filt = filter_signal(sig_filt, fs, 'bandstop', f_range=(40,55), filter_type='iir', butterworth_order=3)
    times = np.arange(0, sig_filt.shape[0])
    # print(times)

    ts = sig_filt

    # plt.plot(np.linspace(0,len(ts), len(ts)), ts)
    # plt.show()

    # freqs, spectra = compute_spectrum(sig_filt, fs)
    # plot_power_spectra(freqs, spectra)
    # freqs, spectra = compute_spectrum(windowed_smoothed_data, fs)
    # plot_power_spectra(freqs, spectra)

    # smoothing for finding control points
    datalen = len(data)
    # https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    data_smoothed = linear_smoothing(sig_filt,H_VAL)
    # print(data_smoothed)
    # plt.plot(times, data_smoothed)



    # %%
    # output= pd.read_csv(index_col=0,filepath_or_buffer= 'gaussianTable.csv')
    # output

    # %%
    # First find R peaks
    peaks = find_peaks(sig_filt, height=10000, distance=500)
    idx_rvals = peaks[0] # spike indices
    amp_rvals = peaks[1]['peak_heights'] # spike amplitudes

    # %%
    # Number of R peaks found
    idx_rvals.shape

    # %%
    # Plotting all windowed cycles to find generalizable window length
    for idx, cycle in enumerate(idx_rvals[1:]):
        #times declaratio nmoved to sig_filt declaration. saves on allocations and increases availability.

        # create window indices
        window_length = (300*fs/1000, 600*fs/1000)  # in ms
        window_pre = int(window_length[0])
        window_post = int(window_length[1])

        # get window
        window_cycle_pre = (cycle-window_pre)
        window_cycle_post = (cycle+window_post)

        # get window for times as well
        windowed_times = times[(cycle-window_pre):
                            (cycle+window_post)]
        windowed_times = np.arange(0, windowed_times.shape[0])

        # get data window
        windowed_data = sig_filt[window_cycle_pre:window_cycle_post]
        windowed_smoothed_data = data_smoothed[window_cycle_pre:window_cycle_post]

        # plt.plot(windowed_times, windowed_data, 'b', alpha=0.2)
        # plt.plot(windowed_times, windowed_data-windowed_smoothed_data, 'r', alpha=0.8)
        # plt.plot(windowed_times, windowed_smoothed_data, 'g', alpha=0.8)
        #plt.plot(cycle, amp_rvals[idx], 'o')

    # plt.show()


    # %%
    idx_pvals = np.zeros(idx_rvals.shape)
    amp_pvals = np.zeros(idx_rvals.shape)
    idx_max = -np.inf
    peak_max = -np.inf

    p = np.zeros(len(idx_rvals))
    q = np.zeros(len(idx_rvals))
    r = np.zeros(len(idx_rvals))
    s = np.zeros(len(idx_rvals))
    t = np.zeros(len(idx_rvals))
    u = np.zeros(len(idx_rvals))
    qs = np.zeros(len(idx_rvals))
    windowed_data_collection = [None] * len(idx_rvals)
    smoothed_windowed_data_collection = [None] * len(idx_rvals)
    rolling_index = 0
    cycle_starts = np.zeros(len(idx_rvals))
    cycle_lengths = np.zeros(len(idx_rvals))
    # failing_cases = [478, 575]
    errstring = 'Failing case: '
    for idx, cycle in enumerate(idx_rvals[:]):
        # print(idx)
        # if idx in failing_cases:
        #     continue
        times = np.arange(0, sig_filt.shape[0])
        # create window indices
        window_length = (300*fs/1000, 600*fs/1000)  # in ms
        window_pre = int(window_length[0])
        window_post = int(window_length[1])
        # get window
        window_cycle_pre = (cycle-window_pre)
        window_cycle_post = (cycle+window_post)
        # get window for times as well
        windowed_times = times[(cycle-window_pre):
                            (cycle+window_post)]
        windowed_times = np.arange(0, windowed_times.shape[0])
        # get data window
        windowed_data = sig_filt[window_cycle_pre:window_cycle_post]
        windowed_smoothed_data = data_smoothed[window_cycle_pre:window_cycle_post]
        
        windowed_data_collection[idx]=windowed_data
        smoothed_windowed_data_collection[idx]=windowed_smoothed_data
        cycle_starts[idx]=window_cycle_pre
        cycle_lengths[idx]=len(windowed_data)

        # was just using
        # plt.plot(windowed_times, sig_filt[window_cycle_pre:window_cycle_post], 'b', alpha=0.8)

        # plt.plot(windowed_times, windowed_data-windowed_smoothed_data, 'r', alpha=0.8)
        # plt.plot(windowed_times, windowed_smoothed_data, 'r', alpha=0.8)

        # is usually enabled, but worked well eneough and became repetitive output... deprecated but probably useful for future testing.
        # plt.plot(windowed_times[p_idx], sig_filt[p_idx], 'o')
        # plt.plot(windowed_times[q_idx], sig_filt[q_idx], 'o')
        # plt.plot(windowed_times[r_idx], sig_filt[r_idx], 'o')
        # plt.plot(windowed_times[s_idx], sig_filt[s_idx], 'o')
        # plt.plot(windowed_times[t_idx], sig_filt[t_idx], 'o')
        # plt.plot(windowed_times[u_idx], sig_filt[u_idx], 'o')
    print("all cycles plotted")

    average_signal = np.zeros(900)
    average_signal = np.nanmean(windowed_data_collection, 0)
    print(average_signal)
    # for i in range(len(average_signal)):
    #     for j in range(len(windowed_data_collection)):
    #         if windowed_data_collection[j][i]==np.nan or windowed_data_collection[j][i]==np.inf or windowed_data_collection[j][i]==-np.inf:
    #             continue
    #         else:
    #             average_signal[i]+=windowed_data_collection[j][i]
    #     average_signal[i]=average_signal[i]/float(len(windowed_data_collection))
    for i in range(np.shape(windowed_data_collection)[0]):
        plt.plot(windowed_times, windowed_data_collection[i], 'r', alpha=0.05)
        result = pearsonr(np.nan_to_num(windowed_data_collection[i], copy=False, nan=0.0, posinf=0.0, neginf=0.0), average_signal)
        print(result)
        #sys.__exit__(3)

    
    
    plt.plot(windowed_times, average_signal, 'b', alpha=1)
    plt.show()
    # plt.draw()

    # print("failing_cases: " + str(failing_cases))

    p_co = np.zeros(len(p))
    q_co = np.zeros(len(p))
    r_co = np.zeros(len(p))
    s_co = np.zeros(len(p))
    t_co = np.zeros(len(p))
    u_co = np.zeros(len(p))

    # asserting here that all the arrays are the same length
    # copies the arrays?
    for i in range(len(p)):
        p_co[i]=windowed_data[int(p[i])]
        q_co[i]=windowed_data[int(q[i])]
        r_co[i]=windowed_data[int(r[i])]
        s_co[i]=windowed_data[int(s[i])]
        t_co[i]=windowed_data[int(t[i])]
        u_co[i]=windowed_data[int(u[i])]



    # %%
    ts = windowed_data_collection[0]

    # plt.plot(np.linspace(0,len(ts), len(ts)), ts)
    # plt.show()

    # freqs, spectra = compute_spectrum(data, fs)
    # plot_power_spectra(freqs, spectra)
    # freqs, spectra = compute_spectrum(windowed_smoothed_data, fs)
    # plot_power_spectra(freqs, spectra)


    # %% [markdown]
    # Gradient Descent Gaussian Stuff

    # %%
    data_for_df = []
    divisors = np.linspace(1, 70, 70)
    smoothing_interval = H_VAL

    # changed or added everything above here
    # print("one cycle plotted alone")
    # plt.plot(np.linspace(0,cycle_lengths[0],cycle_lengths[0]), windowed_data_collection[0], 'r')
    # plt.draw()

    highests = []
    highest_idx = - 1
    smoothing_interval = 15

    # # https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
    # color = iter(cm.rainbow(np.linspace(0, 1, len(divisors))))
    # for i in range(len(divisors)):
    #     # if i==0:
    #     #     plt.plot(np.linspace(0,cycle_lengths[0],cycle_lengths[0]), smoothed_data, 'g', alpha=0.8)
    #     #     continue
    #     smoothing_interval = math.floor(qs[0]/divisors[i])
    #     if smoothing_interval == 0:
    #         smoothing_interval = 1
    #     # print(qs[i], divisors[i], smoothing_interval)
    #     smoothed_data = linear_smoothing(
    #         sig_filt[cycle_starts[0]:cycle_starts[0]+cycle_lengths[0]], smoothing_interval)
    #     if smoothed_data is None:
    #         # print("smoothing failed")
    #         shouldibreak = True
    #         break
    #     highests.append(smoothed_data[650])
    #     # plt.plot(np.linspace(0,cycle_lengths[0],cycle_lengths[0]), smoothed_data, next(color), linewidth=1)

    # print("highests: " + str(highests))
    # plt.draw()

    # %%
    gaussians = []

    for i in range(len(windowed_data_collection)):
        # print(windowed_data_collection[i])
        # if windowed_data_collection[i] is None:
        #     gaussians.append()
        result = fit_n_gaussian_center_bounds(
            x=windowed_data_collection[i],
            b_bounds=[(p[i]-30,p[i]+30),
                    (q[i]-30,q[i]+30),
                    (r[i]-30,r[i]+30),
                    (s[i]-30,s[i]+30),
                    (t[i]-30,t[i]+30)])
        # print(result)
        gaussians.append(result)
    # print(len(gaussians))


    # %%
    gauss_sigs =[None] * len(gaussians)

    print(len(gaussians))

    for i in range(len(gaussians)):
        # print(i)
        gauss_sigs[i]=gaussians[i]
        
    print(len(gauss_sigs))

    df = pd.DataFrame(gauss_sigs).T
    print(df.shape)
    res = df.corr()
    print(res.shape)

    gaussian_corr = np.zeros(df.shape[1])
    for i in range(0, df.shape[1]):
        for j in range(0,df.shape[1]):
            gaussian_corr[j] = gaussian_corr[j] + res[i][j]
        
    score_indices = np.argsort(gaussian_corr)
    score_indices = np.flip(score_indices)

    df.to_csv("gauss_corr_"+str(DATA_ID)+".csv")
    sc = pd.DataFrame(score_indices)
    sc.to_csv("gauss_scores_"+str(DATA_ID)+".csv")

    # sns.heatmap(res, cmap="Greens",annot=True)

    # print(df.shape)
    # res = df.corr()
    # lenres = len(res)
    # # min, min_idx, max, max_idx
    # resdata = [np.zeros(lenres),np.zeros(lenres),np.zeros(lenres),np.zeros(lenres)]

    # %%
    new_sigs =[None] * len(windowed_data_collection)
    results = [None] * len(smoothed_windowed_data_collection)
    # curr_result=[None] * len(smoothed_windowed_data_collection)

    for i in range(len(windowed_data_collection)):
        new_sigs[i]=windowed_data_collection[i]
        
    print(len(new_sigs))

    df = pd.DataFrame(new_sigs)
    df=df.T
    res = df.corr()

    print(df.shape)
    scores = np.zeros(df.shape[1])
    for i in range(0, df.shape[1]):
        for j in range(0,df.shape[1]):
            scores[j] = scores[j] + res[i][j]
        
    score_indices = np.argsort(scores)
    score_indices = np.flip(score_indices)
    print(len(score_indices))
    print(len(gaussians))
    print(len(smoothed_windowed_data_collection))
    print(df.shape)
    print(res.shape)
    # print(scores)
    # print(score_indices)
    # print(len(score_indices))

    for idx in range(0,len(smoothed_windowed_data_collection)):
        results[idx]=[idx,gaussians[idx],score_indices[idx], scores[idx]]

    # df.to_csv("signal_corr_"+str(DATA_ID)+".csv")
    # sc = pd.DataFrame(scores)
    # sc.to_csv("signal_corr_scores_"+str(DATA_ID)+".csv")

    result_df = pd.DataFrame(results)



    # %%

    result_df = result_df
    result_df.to_csv("result_"+str(DATA_ID)+".csv")

    # %%

    plt.show()
    # for i in score_indices[1:10]:
    #     plt.plot(np.linspace(0,len(windowed_data_collection[i]),len(windowed_data_collection[i])), windowed_data_collection[i])

    # #TODO: uncomment block. I commented it out for benchmarking.
    # fig1=plt.figure("best fits")
    # plt.title("plot of best fits, no U-separation yet.")
    # for i in score_indices[0:10]:
    #     plt.plot(np.linspace(0,len(windowed_data_collection[i]),len(windowed_data_collection[i])), windowed_data_collection[i])
    #     # plt.plot(np.linspace(0,len(windowed_data_collection[i]),len(windowed_data_collection[i])), n_skewed_gaussian(*gaussians[i]))

    # fig2=plt.figure("worst fits")
    # plt.title("plots of worst fits, no u separation yet")
    # for i in score_indices[-10:-1]:
    #     #TODO: reinstate top plot. I took it out for 
    #     plt.plot(np.linspace(0,len(windowed_data_collection[i]),len(windowed_data_collection[i])), windowed_data_collection[i])
    #     # plt.plot(np.linspace(0,len(windowed_data_collection[i]),len(windowed_data_collection[i])), n_skewed_gaussian(*gaussians[i]))
    # plt.show()

    # %%

    print(df.shape)
    res = df.corr()
    print(res.shape)
    # sns.heatmap(res, cmap="Greens",annot=True)


    # print(df.shape)
    # res = df.corr()
    # lenres = len(res)
    # # min, min_idx, max, max_idx
    # resdata = [np.zeros(lenres),np.zeros(lenres),np.zeros(lenres),np.zeros(lenres)]

