
def chop_timeserie(data, n_samples):
    timeseries = []
    i = 0 
    while(i * n_samples < data.shape[0]):

        if((i + 1) * n_samples > data.shape[0]):
            timeseries.append(data:)
        timeseries.append(data[i:i+1])
    return timeseries
