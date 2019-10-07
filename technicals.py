import numpy as np
import pandas as pd

# The functions below return the indicator that we use in the strategy,
# eg  ma returns ma-price. For the simpler ma (eg for use in charting),
# use the inbuilt pandas equivalent as used in the bodies of the functions.

def chg(series,d1): 
    """ 
    Compare the absolute price level now with that d1-days ago.
    """
    # prev = series.shift(periods=d1)
    # return series-prev
    return series.diff(periods=d1)

def ret(series,d1): 
    """ 
    Compare the return from price level now with that d1-days ago.
    """
    # prev = series.shift(periods=d1)
    # return (series-prev)/prev
    return series.pct_change(periods=d1)

def ma(series,d1): 
    """ 
    Compare the price level now with current d1-day simple moving average.
    Returns absolute difference (not percentage)
    """
    return series - series.rolling(window=d1).mean()

def ema(series,d1):
    """
    Exponential moving average (data must in increasing date order)
    """
    return series.ewm(span=d1,min_periods=d1, adjust=False).mean()


def xover(series,d1,d2): 
    """ 
    Compare current d1-day simple moving average with current d2-day simple
    moving average. (where d2 > d1)(d1-d2 is bullish)
    """
    return series.rolling(window=d1).mean()-series.rolling(window=d2).mean()

def up2down(series,d1):
    """
    Measure the average positive daily return relative to average negative
    return over the past d1-days; (similar called Gain to Pain indiciator
    except that is based on totals, not averages).
    http://eclecticinvestor.com/the-gain-to-pain-ratio/
    """
    rets=ret(series,1)
    posrets = rets.where(lambda x : x>0)
    negrets = rets.where(lambda x : x<0)*-1

    return posrets.rolling(window=d1,min_periods=1).mean()/negrets.rolling(window=d1,min_periods=1).mean()

def macd(series, d1=12, d2=26, d3=9):
    """
    Calculate the Moving Average Convergence-Divergence indicator: let x be
    the difference between 12-day exponential moving average (EMA) and 26-day
    EMA. MACD compares x with its 9-day EMA.
    """
    exp1 = ema(series, d1)
    exp2 = ema(series, d2)
    macd = exp1-exp2

    exp3 = ema(macd, d3)

    return exp3

def rsi(series,d1):
    """
    Calculate the Relative Strength Indicator: this measure the velocity of
    price movement.
    """
    chgs = chg(series,1)
    u = chgs.where(lambda x : x>0, other=0)
    d = chgs.where(lambda x : x<0, other=0)*-1
    rs = ema(u,d1)/ema(d,d1)
    rsi = 100-100/(1+rs)
    
    return rsi

def ret2vol(series, d1):
    """
    Divide the return of the last d1 days by the volatility of returns
    """
    rets = ret(series,d1)
    vols = series.pct_change().rolling(window=d1).std()*(252**0.5)

    return rets/vols

def invvol(series, d1):
    """
    Calculate the Inverse Volatility indicator: divide 1 by the volatility of
    returns. 
    """
    vols = series.pct_change().rolling(window=d1).std()*(252**0.5)
    return 1/vols

def mvpreg(series):
    """
    Calculate Multi-Variate Panel Regression trend filtering method. 
    """
    pass


