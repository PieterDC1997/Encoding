####################
###IMPORT MODULES###
####################
import os, collections
import os.path as op
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import scale
from numpy.fft import fft2, fftshift, fftfreq
import warnings
warnings.simplefilter('ignore')
#%% SET FILE PATHS
os.getcwd()
os.chdir('C/.../Tutorial')
cwd='C/.../Tutorial'
raw_path='C/.../Tutorial/raw_data/'
save_path='C/.../Tutorial/features/'
# We'll save all timeseries data at this sampling frequency
sfreq_new = 200

#%% ECOG DATA
ecog = mne.read_epochs(raw_path + 'ecog-epo.fif', preload=True)
ecog.plot(scalings='auto', n_epochs=5, n_channels=1, picks=['ch_0'])

#average across all trials, example
ecog_av=ecog.copy()
mne.baseline.rescale(ecog_av._data, ecog.times, (-0.8,-0.1), mode='zscore', copy=False)
_=ecog_av.average().crop(-0.2,0.5).plot()

#filter ecog data with Morlet Wavelets
high_freq = ecog.copy()
freqs = np.logspace(np.log10(70), np.log10(140), 14)
etfr = mne.time_frequency.tfr_morlet(
    high_freq, freqs, picks=range(len(high_freq.ch_names)),
    n_cycles=freqs / 4., average=False, use_fft=True, return_itc=False)
#average across frequencies for a single time-varying amplitude
tfrs = etfr.data.mean(axis=2)

#rescale data
high_freq._data=tfrs.copy()
_ = mne.baseline.rescale(high_freq._data, high_freq.times, (-.8, -.1),
                         mode='zscore', copy=False)
#plot
_ = high_freq.copy().crop(-.5, 1.5).average().plot()

#resample to save computation time / space
high_freq.resample(sfreq_new, npad='auto')
#save to disk
high_freq.save(save_path + 'hfa_ecog-epo.fif', overwrite=True)
#%% AUDIO FILES
#audio files are already split into a trials format (in epochs)
audio = mne.read_epochs(raw_path + 'audio-epo.fif', preload=True)
#add a minimal amount of white noise so that we don't have zeros
audio._data += .1 * np.std(audio._data) * np.random.randn(*audio._data.shape)
#visualize
_ = audio.plot(picks=[0], scalings='auto', n_epochs=5)

#%% 1: SPEECH ENVELOPE
#by taking the absolute value 
envelope = audio.copy()
envelope._data = np.abs(envelope._data)
envelope._data = mne.filter.filter_data(envelope._data,
                                        envelope.info['sfreq'],
                                        None,
                                        10) #filter length?
aba=envelope.resample(sfreq_new)
#plot
_ = envelope.plot(picks=[0], scalings='auto', n_epochs=5)
#save
envelope.save(save_path + 'envelope-epo.fif', overwrite=True)

#%% 2:  SPECTROGRAMS OF SPEECH

frequencies = np.logspace(np.log10(300), np.log10(7000), 64)
n_decim = int(audio.info['sfreq'] // sfreq_new) #to reduce memory usage
tfrs = mne.time_frequency.tfr_morlet(
    audio, frequencies, picks=[0], n_cycles=frequencies / 4., average=False,
    use_fft=True, return_itc=False, decim=n_decim) #decim=to reduce memory usage

#plot
fig, ax = plt.subplots()
mask = mne.utils._time_mask(tfrs.times, -.5, 4)
plt_tfrs = tfrs.data[20].squeeze()[..., mask]
mesh = ax.pcolormesh(tfrs.times[mask], frequencies, np.log(plt_tfrs),
                     cmap='viridis', vmin=10, vmax=20)
ax.axis('tight')
ax.set_title('Spectrogram of speech')
ax.set_ylabel('Frequency Band (Hz)')
ax.set_xlabel('Time (s)')
cbar = fig.colorbar(mesh)
cbar.set_label('Log Amplitude')

#convert it to Epochs so we can use later
info_spec = mne.create_info(['{:.2f}'.format(ifreq) for ifreq in frequencies],
                            sfreq_new, 'misc')
tfrs = mne.EpochsArray(tfrs.data.squeeze(), info_spec, tmin=tfrs.times.min())
#save
tfrs.save(save_path + 'spectrogram-epo.fif', overwrite=True)

#%% 3: PHONEMES
# read in timing information
stimuli = pd.read_csv(raw_path + 'time_info.csv', index_col=0)
sfreq_timit = pd.read_csv(raw_path + 'meta_sub.csv')['sfreq_timit'].values[0]

phonemes = []
for ii, stim_name in enumerate(stimuli['stim_name']):
    stim_name = stim_name.replace('.wav', '')
    with open(raw_path + 'timit_word_info/text_info/{}.PHN'.format(stim_name, 'r')) as f:
        ph_sentence = f.readlines()
        ph_sentence = [jj.replace('\n', '').split(' ') for jj in ph_sentence]
        ph_sentence = pd.DataFrame(ph_sentence)
        ph_sentence['number'] = ii
        phonemes.append(ph_sentence)
phonemes = pd.concat(phonemes)
phonemes.columns = ['start', 'stop', 'phoneme', 'number']

#TIMIT timings are given in samples. Here we divide by sfreq to get seconds.
phonemes[['start', 'stop']] = phonemes[['start', 'stop']].apply(pd.to_numeric)
phonemes[['start', 'stop']] /= sfreq_timit #to get from frequency to time scale

#we need to assign a unique integer to each phoneme
unique_phonemes = phonemes['phoneme'].unique()
phoneme_dict = {iph:ii for ii, iph in enumerate(unique_phonemes)}
phoneme_data = np.zeros([len(audio), len(unique_phonemes), len(tfrs.times)])
# Loop through phonemes and populate our binary data with its timings
for iep, isentence in phonemes.groupby('number'):
    for _, (istt, istp, iph, _) in isentence.iterrows():
        mask = mne.utils._time_mask(tfrs.times, istt, istp)
        phoneme_data[iep, phoneme_dict[iph], mask] = 1
        
# phoneme_dict tells us which row is associated w/ each phoneme
# This inserts the phoneme names into their proper place
phonemes_rev = dict((jj, ii) for ii, jj in phoneme_dict.items())
phonemes_labels = [phonemes_rev[ii] for ii in np.sort(list(phoneme_dict.values()))]

# Turn phoneme data into an EpochsArray for saving
info = mne.create_info(phonemes_labels, sfreq_new, 'misc')
phoneme_data = mne.EpochsArray(phoneme_data, info, tmin=tfrs.times.min())
phoneme_data.save(save_path + 'phonemes-epo.fif', overwrite=True)

#plot
fig, ax = plt.subplots()
ax.pcolormesh(phoneme_data.times, range(len(unique_phonemes)),
              phoneme_data._data[20], cmap=plt.cm.Greys)
ax.set_ylabel('Phoneme ID (a.u.)')
ax.set_xlabel('Time (s)')
ax.set_title('Phoneme presence for one sentence')

#%% 4: WORDS
words = []
for ii, stim_name in enumerate(stimuli['stim_name']):
    stim_name = stim_name.replace('.wav', '')
    with open(raw_path + 'timit_word_info/text_info/{}.WRD'.format(stim_name, 'r')) as f:
        w_sentence = f.readlines()
        w_sentence = [jj.replace('\n', '').split(' ') for jj in w_sentence]
        w_sentence = pd.DataFrame(w_sentence)
        w_sentence['number'] = ii
        words.append(w_sentence)
words = pd.concat(words)
words.columns = ['start', 'stop', 'word', 'number']


words[['start', 'stop']] = words[['start', 'stop']].apply(pd.to_numeric)
words[['start', 'stop']] /= sfreq_timit #to get from frequency to time 

# Now create a categorical variable for each word and populate with 1s
unique_words = words['word'].unique()
word_dict = {iph:ii for ii, iph in enumerate(unique_words)}
word_data = np.zeros([len(audio), len(unique_words), len(envelope.times)])

for iep, isentence in words.groupby('number'):
    for _, (istt, istp, iph, _) in isentence.iterrows():
        mask = mne.utils._time_mask(envelope.times, istt, istp)
        word_data[iep, word_dict[iph], mask] = 1
        print(word_dict[iph])

words_rev = dict((jj, ii) for ii, jj in word_dict.items())
words_labels = [words_rev[ii] for ii in np.sort(list(word_dict.values()))]

# Turn word data into an EpochsArray for saving   
x=['a{}'.format(i) for i in range (0,173)]    
info = mne.create_info(x, sfreq_new, 'misc')
word_data = mne.EpochsArray(word_data, info, tmin=tfrs.times.min())
word_data.save(save_path + 'words-epo.fif', overwrite=True)

#plot
fig, ax = plt.subplots()
ax.pcolormesh(word_data.times, range(len(unique_words)),
              word_data._data[20], cmap=plt.cm.Greys)
ax.set_ylabel('Word ID (a.u.)')
ax.set_xlabel('Time (s)')
ax.set_title('Word presence for one sentence')

#%% 5: WORD ONSET
words = []
for ii, stim_name in enumerate(stimuli['stim_name']):
    stim_name = stim_name.replace('.wav', '')
    with open(raw_path + 'timit_word_info/text_info/{}.WRD'.format(stim_name, 'r')) as f:
        w_sentence = f.readlines()
        w_sentence = [jj.replace('\n', '').split(' ') for jj in w_sentence]
        w_sentence = pd.DataFrame(w_sentence)
        w_sentence['number'] = ii
        words.append(w_sentence)
words = pd.concat(words)
words.columns = ['start', 'stop', 'word', 'number']

words[['start', 'stop']] = words[['start', 'stop']].apply(pd.to_numeric)
words[['start', 'stop']] /= sfreq_timit #to get from frequency to time 
words=words.drop(columns=['word'])
op=words.drop(columns=['stop'])

#a data point every 5ms
times=np.arange(-1.5, 6, 0.005)
times=np.around(times, decimals=3).reshape(-1,1500)
times=times.repeat(29,axis=0).reshape(29,-1,1500)

#create a definition to find last integer of float. Used to round last integer to .005 or .00, to match our sample rate (200 Hz, 5ms per data point)
def nth_digit(digit, n):
   digit = str(digit)
   return int(digit[::-1][n])

for iep, isentence in op.groupby('number'):
    arr=isentence['start'].values
    arr=np.around(arr, decimals=3)
    arr=arr.tolist()
    x=[]
    print(isentence)
    #for-loop: round to .005/.00
    for i in arr:
        if nth_digit(i,0)==1 or nth_digit(i,0)==2 or nth_digit(i,0)==8 or nth_digit(i,0)==9 or nth_digit(i,0)==0:
            i=np.around(i, decimals=2)
        i=str(i)
        if len(i)==5:
            i=list(i)
            i[4]="5"
            i="".join(i)
        i=float(i)
        x.append(i)
    ok=times[iep].reshape(1500)
    ok=ok.tolist()
    #for-loop: give a 1 at the sample point if the word onset is present, 0 if absent. This results in 219 1's (219 word onsets)
    for n, ii in enumerate(ok):
        if ii == 1:
            ok[n] = 250
        if ii in x:
            ok[n] = 1
        if 1 in x:
            for nn, iii in enumerate(ok):
                if ii==250:
                    ok[1]=1
        if ii not in x:
            ok[n] = 0
    times[iep]=ok
channel=['word_onset']
info = mne.create_info(channel, sfreq_new, 'misc')
onset_data = mne.EpochsArray(times, info, tmin=tfrs.times.min())

# Plot the word onsets
plot=plt.plot(onset_data.times, onset_data._data[0][0])
plt.ylabel('Word Onset Presence')
plt.xlabel('Time (s)')

onset_data.save(save_path + 'word_onset-epo.fif', overwrite=True)

#%% SHOWING 1, 2, 3 AND 5 AT THE SAME TIME

fig, axs = plt.subplots(4, 1, sharex=True)
ix = 2
# Plot the envelope
# Note that some values may be negative due to the filtering
axs[0].plot(envelope.times, envelope._data[ix][0])
axs[0].set_title('Speech Envelope')

# Plot the spectrogram
axs[1].pcolormesh(tfrs.times, frequencies, np.log(tfrs._data[ix]),
                  cmap=plt.cm.viridis, vmin=10, vmax=20)
axs[1].set_title('Spectrogram')

# Plot the phoneme labeling
axs[2].pcolormesh(tfrs.times, range(len(unique_phonemes)),
                  phoneme_data._data[ix], cmap=plt.cm.Greys)
axs[2].set_title('Phonemes')
axs[2].set_xlabel('Time (s)')

axs[3].plot(onset_data.times, onset_data._data[ix][0])
axs[3].set_title('Word Onset')
axs[3].set_xlabel('Time (s)')
for ax in axs:
    ax.axis('tight')
plt.tight_layout()

#%% 6: PHONETIC FEATURES
phoneme_labels = pd.read_csv(raw_path + 'phoneme_labels.csv', index_col=0)

###YOU CAN SKIP THE NEXT LINES IF YOU ALREADY RAN CELL 'PHONEMES'###
phonemes = []
for ii, stim_name in enumerate(stimuli['stim_name']):
    stim_name = stim_name.replace('.wav', '')
    with open(raw_path + 'timit_word_info/text_info/{}.PHN'.format(stim_name, 'r')) as f:
        ph_sentence = f.readlines()
        ph_sentence = [jj.replace('\n', '').split(' ') for jj in ph_sentence]
        ph_sentence = pd.DataFrame(ph_sentence)
        ph_sentence['number'] = ii
        phonemes.append(ph_sentence)
phonemes = pd.concat(phonemes)
phonemes.columns = ['start', 'stop', 'phoneme', 'number']
##TIMIT timings are given in samples. Here we divide by sfreq to get seconds.
phonemes[['start', 'stop']] = phonemes[['start', 'stop']].apply(pd.to_numeric)
phonemes[['start', 'stop']] /= sfreq_timit #to get from frequency to time scale
###UNTIL HERE###
phonemes=phonemes[phonemes.phoneme != 'h#']
phns=phonemes['phoneme'].values.tolist()
ftrs=phoneme_labels['phoneme'].values.tolist()
ftrs_=phoneme_labels['kind'].values.tolist()
target=[]

for i in range(len(phns)):
    if phns[i] in ftrs:
        for j in range(len(ftrs)):
            if ftrs[j]==phns[i]:
                target.append(ftrs_[j])
    else:
        target.append('other')
phonemes=phonemes.values 
phonemes=np.column_stack((phonemes, target))
phonemes[:,[2, 4]] = phonemes[:,[4, 2]]
phonemes=pd.DataFrame(phonemes)
phonemes.columns = ['start', 'stop', 'feature', 'number','phonemes']
phonemes=phonemes.drop(columns=['phonemes'])

#we need to assign a unique integer to each phoneme
unique_features = phonemes['feature'].unique()
feature_dict = {iph:ii for ii, iph in enumerate(unique_features)}
feature_data = np.zeros([len(audio), len(unique_features), len(tfrs.times)])
# Loop through phonetic features and populate our binary data with its timings
for iep, isentence in phonemes.groupby('number'):
    for _, (istt, istp, iph, _) in isentence.iterrows():
        mask = mne.utils._time_mask(tfrs.times, istt, istp)
        feature_data[iep, feature_dict[iph], mask] = 1
        
features_rev = dict((jj, ii) for ii, jj in feature_dict.items())
features_labels = [features_rev[ii] for ii in np.sort(list(feature_dict.values()))]

# Turn phoneme data into an EpochsArray for saving
info = mne.create_info(features_labels, sfreq_new, 'misc')
feature_data = mne.EpochsArray(feature_data, info, tmin=tfrs.times.min())
feature_data.save(save_path + 'features-epo.fif', overwrite=True)  

#plot
fig, ax = plt.subplots()
ax.pcolormesh(feature_data.times, range(len(unique_features)),
              feature_data._data[20], cmap=plt.cm.Greys)
ax.set_ylabel('Feature ID (a.u.)')
ax.set_xlabel('Time (s)')
ax.set_title('Phonetic feature presence for one sentence')     
    
#%% SHOWING 1, 2, 3, 5 AND 6 AT THE SAME TIME

fig, axs = plt.subplots(5, 1, sharex=True)
ix = 2
# Plot the envelope
# Note that some values may be negative due to the filtering
axs[0].plot(envelope.times, envelope._data[ix][0])
axs[0].set_title('Speech Envelope')

# Plot the spectrogram
axs[1].pcolormesh(tfrs.times, frequencies, np.log(tfrs._data[ix]),
                  cmap=plt.cm.viridis, vmin=10, vmax=20)
axs[1].set_title('Spectrogram')

# Plot the phoneme labeling
axs[2].pcolormesh(tfrs.times, range(len(unique_phonemes)),
                  phoneme_data._data[ix], cmap=plt.cm.Greys)
axs[2].set_title('Phonemes')
axs[2].set_xlabel('Time (s)')

axs[3].pcolormesh(tfrs.times, range(len(unique_features)),
                  feature_data._data[ix], cmap=plt.cm.Greys)
axs[3].set_title('Phonemetic Features')
axs[3].set_xlabel('Time (s)')

axs[4].plot(onset_data.times, onset_data._data[ix][0])
axs[4].set_title('Word Onset')
axs[4].set_xlabel('Time (s)')
for ax in axs:
    ax.axis('tight')
plt.tight_layout()

#%% 7: YOU CAN TRY THE MODULATION POWER SPECTRUM AS WELL


#%% EPOCHING ECOG BASED ON THE PHONEMES FOR DECODING PURPOSES
"""Finally, in order to perform decoding we need to collect the ECoG activity in response to each phoneme. 
We can simply do this by epoching the ECoG data using the phoneme timings that we created above. 
We can then calculate the response of each electrode for each phoneme.
This will allow us to build a model with ECoG activity as an input, and a phoneme ID as an output. 
Note that this removes information about how the electrode activity changes across time, 
and instead summarizes each epoch with the mean activity per electrode in a short window after each phoneme 
(we'll do this in the model fitting notebook)."""
phone_lengths = np.sort(phonemes['stop'] - phonemes['start'])[::-1]
# We'll take a fixed window after each phoneme's onset
# The length of this window will be the mean length of all phonemes
time_keep_phone = np.mean(phone_lengths)
n_ixs_keep = int(time_keep_phone * high_freq.info['sfreq'])
# Finally, we'll create an Epoched version of *each* phoneme for classification
epochs_ph = []
for ii, phns in phonemes.groupby('number'):
    for _, (phst, phstp, phn, _) in phns.iterrows():
        ix_stt = int(phst * high_freq.info['sfreq'])
        i_mask_time = np.zeros(high_freq._data.shape[-1], dtype=bool)
        i_mask_time[ix_stt:ix_stt + n_ixs_keep] = True
        epochs_ph.append((phn, high_freq._data[ii][..., i_mask_time]))

phones, epochs_ph = zip(*epochs_ph)
phones = np.hstack(phones)
phones_labels = [phoneme_dict[ii] for ii in phones]
epochs_ph = np.stack(epochs_ph)

# Create an MNE representation of these event onsets.
events_phones = np.vstack([np.arange(len(phones_labels)),
                           np.zeros_like(phones_labels),
                           phones_labels]).T
# Now turn into an MNE object
epochs_phones = mne.EpochsArray(epochs_ph, high_freq.info, events_phones, event_id=phoneme_dict, tmin=0)

#(average)
_ = epochs_phones['aa'].average().plot()

#save
epochs_phones.save(save_path + 'ecog_phonemes-epo.fif', overwrite=True)