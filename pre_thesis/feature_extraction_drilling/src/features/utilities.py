
def chop_timeseries(data, n_samples):
    timeseries = []
    i = 0 
    while(i * n_samples < data.shape[0]):
        if((i + 1) * n_samples > data.shape[0]):
            timeseries.append(data[i * n_samples:])
        else:
            timeseries.append(data[i * n_samples:(i + 1) * n_samples])
        
        i += 1
    return timeseries