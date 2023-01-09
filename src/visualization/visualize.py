# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def scaling_comparison(data, scaled, outpath):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,5))    
    ax[0].plot(data, color="blue")
    ax[1].plot(scaled, color="green")
    fig.savefig(outpath)
    plt.show()
    
def scaling_histogram_comparison(data, scaled, outpath):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,5), tight_layout=True)    
    ax[0].hist(data, bins=20, color="blue")
    ax[1].hist(scaled, bins=20, color="green")
    fig.savefig(outpath)
    plt.show()