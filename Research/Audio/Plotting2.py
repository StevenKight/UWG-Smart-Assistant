import os

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
from pylab import *

def f(filename, plotfile):
    fs, data = wavfile.read(filename) # load the data

    a = data.T[0] # this is a two channel soundtrack, I get the first track
    b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
    c = fft(b) # create a list of complex number
    d = int(len(c)/2)  # you only need half of the fft list

    plt.plot(abs(c[:(d-1)]), 'r')
    savefig(plotfile, bbox_inches='tight')

files = os.listdir('Recognition/Audio/Dataset/')

if '.DS_Store' in files:
    files.pop(0)

for file in files:
    plot_file = 'Recognition/Audio/Plots/' + file.replace('wav', 'png')
    file = 'Recognition/Audio/Dataset/' + file

    f(file, plot_file)