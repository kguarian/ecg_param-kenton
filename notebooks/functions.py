

from cmath import isinf
import os
from tqdm.notebook import tqdm

import math
import threading

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

from py_files.extract import extract_data, extract_metadata
from bycycle import features, cyclepoints, plts
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.fft import fft
from neurodsp.filt import filter_signal
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.spectral import compute_spectrum

import statsmodels.api as sm

# Import filter function
from neurodsp.filt import filter_signal

import bycycle

from matplotlib.pyplot import cm

from IPython.display import display

# 25, 5, 10, 15
# 10 currently works best, sample size: 1
# data 0: 10
H_VAL = 10
DATA_ID = 0
# was 1
FREQ_HIGHPASS = 1

# do better?
default_a = [2000, -2000, 2000, -2000, 2000]
default_b = [100, 200, 300, 400, 500]
default_c = [-0.001, -0.001, -0.001, -0.001, -0.001]
default_d = [0, 0, 0, 0, 0]
default_guess_a, default_guess_c, default_guess_d = 2000, -0.001, 0

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
    while initial_derivative == 0:
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
        y2 = time_series[i+remainder]
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
    print(len(guesses))
    return guesses

# inverse to flatten_guesses


def build_guesses(width, *params):
    retVal = []
    if len(params) % width != 0:
        print("length "+str(len(params)) + " is not divisible by " + str(width))
        return None
    for i in range(width):
        retVal.append([])
    for i in range(len(params)):
        retVal[i % width].append(params[i])
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


def CURVE_FIT_5_skewed_gaussian(x, *params):
    newParams = build_guesses(4, *params)
    return n_skewed_gaussian(x, *newParams)

# recursively prints shapes of lists. remove


def print_list_shape(list):
    shadow = list
    try:
        while len(shadow) > 0:
            print(len(shadow))
            shadow = shadow[0]
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

    if len(b_bounds[0]) != len(b_bounds[1]):
        return "bounds have different lengths"

    sz = len(b_bounds)
    a, c, d = [], [], []
    b = []
    a_guesses = []
    b_guesses = []
    c_guesses = []
    d_guesses = []
    guesses = None

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
    print(len(a_guesses))
    print(len(b_guesses))
    print(len(c_guesses))
    print(len(d_guesses))
    guesses = flatten_guesses(a_guesses, b_guesses, c_guesses, d_guesses)
    print_list_shape(guesses)
    for i in range(len(bounds[0])):
        if bounds[0][i] >= bounds[1][i]:
            print("bound at index " + str(i)+" violates le/gt relationship")
    print("bounds: "+str(bounds))
    print("guesses: "+str(guesses))
    # generalized fourier series should replace this funcall
    params = curve_fit(f=CURVE_FIT_5_skewed_gaussian, xdata=np.linspace(0, len(x), len(x)),
                       ydata=x,
                       p0=guesses,
                       #    bounds=bounds,
                       method='lm')
    # X0 must be one-dimensional...let's flatten the array.
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


def find_intercepts_telescoping(time_series, tgt, left_b, right_b, center):
    left = []
    right = []

    left_b = int(left_b)
    right_b = int(right_b)
    if left_b < 0 or right_b < 0 or left_b > right_b:
        return (left, right)
    if right_b > len(time_series):
        right_b = len(time_series)
    for i in range(center-left_b):
        last = time_series[center-i]
        this = time_series[center-i-1]
        # print("first loop: last="+str(last)+", this="+str(this))
        if last > tgt and this <= tgt:
            left.append(int(center-i-1))
            break
        if last <= tgt and this > tgt:
            left.append(int(center-i-1))
            break
    for i in range(right_b-center):
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
