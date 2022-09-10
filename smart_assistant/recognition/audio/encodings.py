import csv
import json
import os

import numpy as np
import scipy.io.wavfile as wavfile

OSC_CI = []
SPEC_CI = []

def oscillogram_properties(file: str) -> dict:
    _, audioBuffer = wavfile.read(file)

    sumList = []
    for index in range(len(audioBuffer)):
        audioBuffer[index] = audioBuffer[index].tolist()
        sumList.append(abs(audioBuffer[index][0]))
    
    n = len(sumList)
    mean = sum(sumList) / n

    sd = np.sqrt(sum((sumList - mean)**2) / (n - 1))
    skew = sum((sumList - mean)**3) / ((n - 1) * (sd ** 3))
    kurt = sum((sumList - mean)**4) / ((n - 1) * (sd ** 4))

    z = 1.960   # Z-Score based on 95% confidence
    ci = (np.around(mean - z * (sd / np.sqrt(n)), 2), np.around(mean + z * (sd / np.sqrt(n)), 2))

    OSC_CI.append(ci)

    result_d = {
        'mean': np.around(mean, 2),
        #'mode': np.around(mode, 2),
        'sd': np.around(sd, 2),
        'skew': np.around(skew, 2),   # Degree of skewness
        'kurt': np.around(kurt, 2),    # Kurtosis, tail weight
        'Confidence interval': ci
    }

    list_d = [
        np.around(mean, 2), np.around(sd, 2), np.around(skew, 2),
        np.around(kurt, 2), ci
    ]

    return result_d, list_d


def spectral_properties(audio: np.ndarray, fs: int) -> dict:
    """https://stackoverflow.com/a/54621604"""
    spec = np.abs(np.fft.rfft(audio))
    freq = np.fft.rfftfreq(len(audio), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    n = len(spec)

    skew = ((z ** 3).sum() / (n - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (n - 1)) / w ** 4

    z = 1.960   # Z-Score based on 95% confidence
    ci = (np.around(mean - z * (sd / np.sqrt(n)), 2), np.around(mean + z * (sd / np.sqrt(n)), 2))

    SPEC_CI.append(ci)

    result_d = {
        'mean': np.around(mean, 2),
        'mode': np.around(mode, 2),
        'sd': np.around(sd, 2),
        'skew': np.around(skew, 2),   # Degree of skewness
        'kurt': np.around(kurt, 2),    # Kurtosis, tail weight
        'Confidence interval': ci
    }

    list_d = [
        np.around(mean, 2), np.around(mode, 2), np.around(sd, 2),
        np.around(skew, 2), np.around(kurt, 2), ci
    ]

    return result_d, list_d

names = next(os.walk('recognition/Audio/Dataset'))[2]

people = {}
cols = ["Frequency mean",
            "Frequency mode",
            "Frequency sd",
            "Frequency skew",
            "Frequency kurt",
            "Frequency Confidence Interval Low",
            "Frequency Confidence Interval High",
            "Amplitude mean",
            "Amplitude sd",
            "Amplitude skew",
            "Amplitude kurt",
            "Amplitude Confidence Interval Low",
            "Amplitude Confidence Interval High",
            "Name"]
rows = []
statistics = {}
for name in names:
    currentFile = 'recognition/Audio/Dataset/' + name
    #files = os.listdir(currentFile)
    if name == ".DS_Store":
        continue

    #for file in files:
    # create data
    Fs, aud = wavfile.read(currentFile)

    statistics["Frequency"], frequency  = spectral_properties(aud[:,0], Fs)
    statistics["Amplitude"], amplitude = oscillogram_properties(currentFile)
    
    row = []
    for stat in frequency:
        if type(stat) == tuple:
            for value in stat:
                row.append(value)
        else:
            row.append(stat) 
    for stat in amplitude:
        if type(stat) == tuple:
            for value in stat:
                row.append(value)
        else:
            row.append(stat) 

    row.append(name.replace(".wav", "").replace(" ","-"))
    rows.append(row)

    people[name] = statistics

try:
    os.remove('smart_assistant/recognition/Audio/data.json')
    os.remove('smart_assistant/recognition/Audio/data.csv')
except FileNotFoundError:
    pass

with open('smart_assistant/recognition/audio/Models/data.json', 'w') as fp:
    json.dump(people, fp, indent=4)

with open('smart_assistant/recognition/audio/Models/data.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(cols)
    write.writerows(rows)

# Create test file data
rows = []
for i in range(1,2):
    Fs, aud = wavfile.read(f'smart_assistant/recognition/Audio/Test/Test_{i}.wav')

    statistics["Frequency"], frequency  = spectral_properties(aud[:,0], Fs)
    statistics["Amplitude"], amplitude = oscillogram_properties(f'smart_assistant/recognition/Audio/Test/Test_{i}.wav')

    row = []
    for stat in frequency:
        if type(stat) == tuple:
            for value in stat:
                row.append(value)
        else:
            row.append(stat) 
    for stat in amplitude:
        if type(stat) == tuple:
            for value in stat:
                row.append(value)
        else:
            row.append(stat) 
    
    row.append(f'Test_{i}')
    rows.append(row)

with open('smart_assistant/recognition/Audio/Models/test.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(cols)
    write.writerows(rows)