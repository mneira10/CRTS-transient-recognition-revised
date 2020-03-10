import math
import numpy as np
import datetime as datetime
from astropy.table import Table
import sncosmo
from lmfit.models import SkewedGaussianModel
from astropy.timeseries import LombScargle

def __datetime_diff_to_int_timedelta__(ss_datetime_diff):
    '''
    Convert datetime series to integer timedelta
    '''
    return ss_datetime_diff.dt.total_seconds() / 3600

def __mag_to_flux__(ss_mag):
    return 10**((-ss_mag+24.80)/2.5)

def magerr_to_fluxerr(m,sigmaM):
    dgdm = (np.log(10)*10**((24.80-m)/2.5)/2.5)
    return (dgdm**2*sigmaM**2)**0.5


def __flux_to_mag__(ss_flux):
    return -np.log10(ss_flux)

def __percentile_diff_ratio__(ss_flux, p):
    '''
    Calculate ratio of flux percentiles (p1 - p2) / (95th - 5th).
    p1: Percentile p. p2: Percentile 100-p.
    '''
    num = ss_flux.quantile(p) - ss_flux.quantile(1.-p)
    denom = ss_flux.quantile(.95) - ss_flux.quantile(.05)
    return num/denom

def __stetson_sigmas__(ss_mag, ss_magerr):
    '''
    Calculates the relative errors (sigmas) for stetson measurements.
    '''
    n = ss_mag.shape[0]
    sigmas = np.sqrt(float(n) / (n - 1)) * (ss_mag - ss_mag.mean()) / ss_magerr
    return sigmas

def skew(ss_mag):
    '''
    Skewness of the magnitudes.
    '''
    return ss_mag.skew()

def kurtosis(ss_mag):
    '''
    Kurtosis of the magnitudes, reliable down to a small number of epochs.
    '''
    return ss_mag.kurtosis()

def small_kurtosis(ss_mag):
    '''
    Small sample kurtosis of the magnitudes.
    See http://www.xycoon.com/peakedness_small_sample_test_1.htm
    '''

    n = float(ss_mag.shape[0])
    mean = ss_mag.mean()
    s = math.sqrt((ss_mag - mean).pow(2).sum() / (n-1))
    S = math.pow( (ss_mag-mean).divide(s).sum(), 4)
    c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return c1 * S - c2

def std(ss_mag):
    '''
    Standard deviation of the magnitudes.
    '''
    return ss_mag.std()

def beyond1st(ss_mag, ss_magerr):
    '''
    Percentage of photometric magnitudes beyond one st. dev. from the weighted
    (by photometric errors) mean.
    '''
    n = float(ss_mag.shape[0])
    weighted_mean = (ss_mag / ss_magerr).sum() /  (1./ss_magerr).sum()
    stdev = std(ss_mag)
    ss_beyond1st = ss_mag[(ss_mag - weighted_mean).abs() > stdev]
    return ss_beyond1st.shape[0] / n

def stetson_j(ss_mag, ss_magerr, ss_date, exponential=False):
    '''
    The Welch-Stetson J variability index (Stetson 1996).
    A robust standard deviation.
    Optional exponential weighting scheme (Zhang et al. 2003) taking successive pairs in time order.
    NOTE: ss_flux must be ordered by it's corresponding date in ss_date.
    '''
    n = float(ss_mag.shape[0])
    if n <= 1: return 0
    # Calculate sigmas: Relative Errors
    sigmas = __stetson_sigmas__(ss_mag, ss_magerr)
    # Calculate weights
    w = np.ones(int(n)); w[0] = 0
    if exponential:
        # Calculate mean dt: delta-time
        dt = __datetime_diff_to_int_timedelta__(ss_date.diff()).mean()
        # Re-calculate Weights
        w = np.exp(-__datetime_diff_to_int_timedelta__(ss_date.diff()) / dt)
    # Calculate p: product of residuals
    p = sigmas * sigmas.shift(1)
    # Return Stetson J measuerement
    return (w * np.sign(p) * p.abs().pow(1./2)).sum() / w.sum()

def stetson_k(ss_mag, ss_magerr):
    '''
    Welch-Stetson variability index K (Stetson 1996).
    Robust kurtosis measure.
    '''
    n = ss_mag.shape[0]
    if n <= 1: return 0
    # Calculate sigmas: Relative Errors
    sigmas = __stetson_sigmas__(ss_mag, ss_magerr)
    # Return Stetson K measurement
    return sigmas.abs().mean() / np.sqrt(sigmas.pow(2.).mean())

def max_slope(ss_mag, ss_date):
    '''
    Maximum absolute magnitude slope between two consecutive observations.
    '''
    ss_timedelta = __datetime_diff_to_int_timedelta__(ss_date.diff())
    return (ss_mag.diff() / ss_timedelta).abs().max()

def amplitude(ss_mag):
    '''
    Half the difference between the maximum and the minimum magnitudes.
    '''
    return ( ss_mag.max() - ss_mag.min() ) / 2.

def median_absolute_deviation(ss_mag):
    '''
    Median discrepancy of the fluxes from the median flux.
    '''
    return ss_mag.mad()

def median_buffer_range_percentage(ss_flux):
    '''
    Percentage of fluxes within 10% of the amplitude from the median.
    '''
    N = ss_flux.shape[0]
    median = ss_flux.median()
    mbf_list = ss_flux[(ss_flux - median).abs() < .1 * median]
    return mbf_list.shape[0] / float(N)

def pair_slope_trend(ss_mag, ss_date):
    '''
    Percentage of all pairs of consecutive mag measurements that have positive slope.
    '''
    N = float(ss_mag.shape[0])
    ss_timedelta = __datetime_diff_to_int_timedelta__(ss_date.diff())
    ss_slopes = (ss_mag.diff() / ss_timedelta)
    return ss_slopes[ss_slopes > 0].shape[0] / N

def pair_slope_trend_last_30(ss_mag, ss_date):
    '''
    Percentage last 30 pairs of consecutive mag measurements that have positive slope.
    '''
    N = 30.
    ss_timedelta = __datetime_diff_to_int_timedelta__(ss_date.diff())
    ss_slopes = (ss_mag.diff().tail(30) / ss_timedelta.tail(30))
    return ss_slopes[ss_slopes > 0].shape[0] / N

def flux_percentile_ratio_mid20(ss_flux):
    '''
    Ratio of flux percentiles (60th - 40th) over (95th - 5th).
    '''
    return __percentile_diff_ratio__(ss_flux, .6)

def flux_percentile_ratio_mid35(ss_flux):
    '''
    Ratio of flux percentiles (67.5th - 32.5th) over (95th - 5th).
    '''
    return __percentile_diff_ratio__(ss_flux, .675)

def flux_percentile_ratio_mid50(ss_flux):
    '''
    Ratio of flux percentiles (75th - 25th) over (95th - 5th).
    '''
    return __percentile_diff_ratio__(ss_flux, .75)

def flux_percentile_ratio_mid65(ss_flux):
    '''
    Ratio of flux percentiles (82.5th - 17.5th) over (95th - 5th).
    '''
    return __percentile_diff_ratio__(ss_flux, .825)

def flux_percentile_ratio_mid80(ss_flux):
    '''
    Ratio of flux percentiles (90th - 10th) over (95th - 5th).
    '''
    return __percentile_diff_ratio__(ss_flux, .9)

def percent_amplitude(ss_flux):
    '''
    Largest percentage difference between either the max or min magnitude and the median.
    '''
    median = ss_flux.median()
    top_diff = abs(ss_flux.max() - median)
    btm_diff = abs(ss_flux.min() - median)
    return np.maximum(top_diff, btm_diff) / median

def percent_difference_flux_percentile(ss_flux):
    '''
    Ratio of (95th - 5th) flux percentile over the median flux.
    '''
    return ( ss_flux.quantile(.95) - ss_flux.quantile(.5) ) / ss_flux.median()

def linear_trend(ss_flux, ss_date):
    '''
    Slope of a linear fit to the light curve fluxes.
    '''
    ss_date_diff = ss_date - datetime.datetime(1970,1,1)
    ss_timedelta = __datetime_diff_to_int_timedelta__(ss_date_diff)
    m, b = np.polyfit(ss_timedelta, ss_flux, 1)
    return m

def poly_params(ss_mag, ss_magerr, ss_mjd):
    '''
    Returns poly fit parameters up to rank 4.
    '''
    x = ss_mjd - ss_mjd.mean(); y = ss_mag 
    p1 = np.polyfit(x, y, 1)
    p2 = np.polyfit(x, y, 2)
    p3 = np.polyfit(x, y, 3)
    p4 = np.polyfit(x, y, 4)
    return p1, p2, p3, p4

def chi2SALT2(df):
    # add extra columns needed by sncosmo
    df = df.copy()
    
    df['zp'] = 24.80
    df['zpsys'] = 'ab'
    df['band'] = 'sdssr'
    
    # rename mjd column to time
    allCols = list(df.columns)
    allCols[2] = 'time'
    df.columns = allCols
    
    #remove magnitudes to avoid errors
    df = df.drop(['Mag','Magerr','Date'],axis=1)
    
    #convert to astropy table
    table = Table.from_pandas(df.reset_index())
    
    #create SALT2 model
    model = sncosmo.Model(source='salt2')
    
    #try to fit model to data, else return -100
#     print(table.columns)
#     print(table)
#     print(table['time'])
#     print(table['Date'])
    try:
        res, fitted_model = sncosmo.fit_lc(table, model, ['z', 't0', 'x0', 'x1', 'c'],  bounds={'z':(0.1, 1.3)})
        return res.chisq
    
    except:
        return -100

def chi2sGauss(df):
    
    # add extra columns needed by sncosmo
    df = df.copy()
    
    df['zp'] = 24.80
    df['zpsys'] = 'ab'
    df['band'] = 'sdssr'
    
    # rename mjd column to time
    allCols = list(df.columns)
    allCols[2] = 'time'
    df.columns = allCols
    
    #remove magnitudes to avoid errors
    df = df.drop(['Mag','Magerr'],axis=1)
    
    #convert to astropy table
    table = Table.from_pandas(df.reset_index())
    
    #try to fit skewed gaussian, else return -100
    try:
        return chiGaussianFromTable(table)
    except:
        return -100

def chiGaussianFromTable(table):
    min_date = min(table["time"])
    max_date = max(table["time"])
    
    data=table
    
    x = data["time"]-min_date
    y = data["Flux"]


    max_i = get_max_index(data["Flux"])
    mean_date = data[max_i]["time"]-min_date


    n = len(x)
    sigma = np.std(y)

    
    model = SkewedGaussianModel()

    # set initial parameter values
    params = model.make_params(amplitude=max(y), center=mean_date, sigma=sigma, gamma=1)

    # adjust parameters  to best fit data.
    result = model.fit(y, params, x=x)

    return result.chisqr


def get_max_index(fluxes):
    max_val = 0
    max_index = -1
    for i, elem in enumerate(fluxes):
        
        if max_val<elem:
            max_val = elem
            max_index = i
            
    return max_index

def magnitudeRatio(df):
    '''
    P(mag>median(mag))
    i.e. number of points in the curve above median mag / total no points
    Note that median mag of entire dataset, otherwise always = 0.5
    '''
    #precalc on entire dataset (with oversample)
    globalMedian = 18.976919579290353
    
    return np.sum(df.Mag>globalMedian)/len(df)

def lombScargle(df):
    '''
    Taken from:
    https://docs.astropy.org/en/stable/timeseries/lombscargle.html
    
    frequency is in Hz. It is not w (angular freq) => T = 1/f
    
    from disanto paper: 
    m (ls): the period obtained by the peak frequency of the Lomb–Scargle periodogram
    '''
    frequency, power = LombScargle(df.MJD, df.Mag,df.Magerr).autopower()
    max_frequency = frequency[np.argmax(power)]
    return 1/max_frequency
