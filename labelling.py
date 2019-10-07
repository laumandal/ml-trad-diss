import pandas as pd
import numpy as np
from tqdm import tqdm

def triple_barrier_label(px, upper=None, lower=None, timeout=None):
    """
    Take in the prices, 3 difference barriers, and return the labels
    Arguments:
        px: pandas series of prices (indices must be datetimes, in increasing order)
        upper: take profit level in pct
        lower: stop loss level in pct
        timeout: timeout as a datetime delta
    Returns:
        out: a df of outputs
    """
    
    #create the dataframe to return
    out = pd.DataFrame(px)
    
    if timeout is None:
        # set timeout to be the last timestamp of the series
        out['timeout']=px.index[-1]
    else:
        out['timeout']=px.index+timeout
        
    # TODO: consider an easier way to have long/short
    # by setting side=1 we only consider long here
    # this will work fine for equal stop/tp as we will predict upper/lower
    # barrier and can act accordingly
    side = 1
    
    for periodstart, periodend in tqdm(out.timeout.iteritems(), 
                                       total=len(out.index),
                                       desc='labelling'):
        # take the period for the trade
        period = px[periodstart:periodend]
        # convert prices to returns since trade started
        period = (period/px[periodstart]-1)*side
        # find the first date that return is less than the lower barrier
        out.loc[periodstart,'lower'] = period[period<lower*-1].index.min()
        # find the first date that return is more than the upper barrier
        out.loc[periodstart,'upper'] = period[period>upper].index.min()
    
    #out['label'] = out[['lower','timeout','upper']].idxmin(axis=1)
    
    #to get min as labels, convert to numpy, get min, take 1 off
    # columns must be selected in label order 
    out['label'] = np.argmin(out[['lower','timeout','upper']].values, axis=1) -1
    
    return out