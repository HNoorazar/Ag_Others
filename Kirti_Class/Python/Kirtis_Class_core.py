import numpy as np
import pandas as pd
# import geopandas as gpd
from IPython.display import Image
# from shapely.geometry import Point, Polygon
from math import factorial
import datetime
import time
import scipy
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb

import os, os.path
import math
import sys

###########################################################


def create_cosine_arr(range_list):
    """Return 2*pi*x where x runs from range_list[0] to range_list[1]
    Arguments
    ---------
    range_list : an list of length 2
        left and right end points of a line segments

    Returns
    -------
    sequence : list like of floats
         sequence from range_list[0] to range_list[1] with step size 0.02

    cosine_list : list like of floats
        An array with values of 2*pi*x where x's are values lying between
        range_list[0] and range_list[1] in increments of 0.02
    """
    sequence = np.arange(range_list[0], range_list[1], 0.02)
    return ([sequence, np.cos(2*np.pi*sequence)])



def diversify(prices_arr, portfolio_size, invest_fund=1, preferred=None):
    """Return (stocks indices, weights [norm of eita - eita bar]^-1) for a diversified portfolio
    Arguments
    ---------
    prices_arr : array like
        Stock prices: `prices_arr.shape = (Ndays, Nstocks)`
    portfolio_size : int
        Number of diverse stocks to choose.
    preferred : list of int
        List of preferred vectors. These columns should be in the
        diversified return

    Returns
    -------
    weights_dict : dict(ind:weight)
        Dictionary of recommended diversification. Keys are the indices,
        and the values are the associated weights.
    stocks : array
        Recommended list of `portfolio_size` stocks for a diversified portfolio
    weights : array_like
        Array of weights of investment for each stock

    Example
    -------
    >>> prices_arr = np.arange(1, 1251).reshape((250, 5)) # for plotting
    >>> diversify(prices_arr=prices_arr, portfolio_size=3, invest_fund=1000)
    OrderedDict([(3, 450.15...), (1, 312.327...), (0, 237.52...)])
    """


