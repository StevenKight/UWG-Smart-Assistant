import os

# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
import scipy.io
import scipy.io.wavfile
# for opening the media file
import scipy.io.wavfile as wavfile
import sys

files = os.listdir('recognition/Audio/Dataset/')

if '.DS_Store' in files:
    files.pop(0)

for file in files:
    # making plots
    fig, ax = plt.subplots(2, 1)

    plot_file = 'recognition/Audio/Plots/' + file.replace('wav', 'png')
    file = 'recognition/Audio/Dataset/' + file

    # create data
    Fs, aud = wavfile.read(file)
    
    """Oscillogram (plot 1):"""
    sampleRate, audioBuffer = scipy.io.wavfile.read(file)
    duration = len(audioBuffer)/sampleRate
    time = np.arange(0,duration,1/sampleRate) #time vector

    ax[0].set(ylabel='Amplitude')
    ax[0].plot(time, audioBuffer)
    ax[0].set_xticklabels([])

    """Spectragram (plot 2):"""
    # select left channel only
    aud = aud[:,0]

    ax[1].set(xlabel='Time', ylabel='Frequency')
    ax[1].specgram(aud, Fs=Fs)

    # set the spacing between subplots
    fig.tight_layout()
    plt.savefig(plot_file)
