import numpy as np

def findQs(detuning, stepSize=5E-5, maxJ=18.0, dJ=0.1, margin=1):
    '''
    '''
    myMin = 1.0
    myMax = 0.0
    for jx in np.arange(0, maxJ, dJ):
        for jy in np.arange(0, maxJ, dJ):
            value = detuning(jx, jy)
            if value < myMin :
                myMin = value
            if value > myMax :
                myMax = value
    return np.arange(myMin-margin*stepSize, myMax+margin*stepSize, stepSize)

def get_tune_range(detuning, maxJ=18.0, margin=1e-4, n_samples=100):
    myMin = 1.0
    myMax = 0.0
    for jx in np.linspace(0, maxJ, 100):
        for jy in np.linspace(0, maxJ, 100):
            value = detuning(jx, jy)
            if value < myMin :
                myMin = value
            if value > myMax :
                myMax = value
    return np.linspace(myMin - margin, myMax + margin, n_samples)
    
    
def get_tune_range_focused(detuning, maxJ=18.0, margin=1e-4, n_samples=100,
                           center_tune = None , width_ratio=1):
    """
        center_tune: tune around which SD is centered
        width_ratio: ratio of width of focused SD to width of full SD
    """
    full_tune_range = get_tune_range(detuning, maxJ=18.0, margin=1e-4, n_samples=2)
    full_tune_width = full_tune_range[-1]-full_tune_range[0]
    minQ = center_tune - 0.5*full_tune_width*width_ratio
    maxQ = center_tune + 0.5*full_tune_width*width_ratio
    if width_ratio<0:
        return np.linspace(maxQ,minQ,n_samples)
    else:
        center_tune_ratio = (center_tune-full_tune_range[0])/full_tune_width  # how long from a to b
        lower_width_ratio = max(0,center_tune_ratio-0.5*width_ratio)
        
        n_samples_lower  = int(n_samples/2*lower_width_ratio)
        n_samples_upper  = int(n_samples/2-n_samples_lower)
        n_samples_inner  = n_samples -n_samples_lower-n_samples_upper
                
        return np.concatenate((np.linspace(full_tune_range[0] ,minQ,n_samples_lower,endpoint=False),
                               np.linspace(minQ               ,maxQ,n_samples_inner,endpoint=False), 
                               np.linspace(maxQ,full_tune_range[-1],n_samples_upper)))

    
