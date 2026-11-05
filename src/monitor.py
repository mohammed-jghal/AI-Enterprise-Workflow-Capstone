"""Distribution drift monitoring using the Wasserstein distance."""
import numpy as np
from scipy.stats import wasserstein_distance

def drift_distance(reference, current):
    a=np.asarray(reference, dtype=float); b=np.asarray(current, dtype=float)
    if a.size==0 or b.size==0: raise ValueError('Both distributions must contain values')
    return float(wasserstein_distance(a,b))

def monitor(reference, current, threshold=250.0):
    """Wasserstein drift check.

    ``threshold`` is in revenue units (same units as the distributions)
    and is an uncalibrated default: pick it from the reference
    distribution's own scale (e.g. a multiple of its standard
    deviation) before using drift decisions to trigger retraining.
    """
    distance=drift_distance(reference,current)
    return {'distance':distance,'threshold':float(threshold),'drift':distance>threshold,'action':'retrain' if distance>threshold else 'continue'}

if __name__=='__main__':
    print(monitor([100,110,105],[105,115,108]))
