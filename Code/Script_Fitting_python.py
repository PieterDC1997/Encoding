####################
###IMPORT MODULES###
####################
import os
import os.path as op
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import modelingtools
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.preprocessing import StandardScaler, scale
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from modelingtools import delay_time_series, plot_activity_on_brain
from IPython.html.widgets import interact, fixed
import warnings

#%% SET FILE PATHS
os.getcwd()
os.chdir('C/.../Tutorial')
cwd='C/.../Tutorial'
raw_path='C/.../Tutorial/raw_data/'
feature_path='C/.../Tutorial/features/'
#%%ECOG, DV IN ENCODING MODELS
ecog = mne.read_epochs(feature_path + 'hfa_ecog-epo.fif', preload=True)
mask_time_ecog = mne.utils._time_mask(ecog.times, -.5, 3.5) #

# In this tutorial, we'll find the electrode with the largest evoked response and select it as our model output
activity = ecog.copy()
_ = mne.baseline.rescale(activity._data, activity.times, (None, 0), mode='zscore', copy=False)
activity = activity._data.mean(0).mean(-1)
ix_active=np.argmax(activity) #this finds the electrode with highest activity: electrode 32 (31 python count)

#visualization
fig, ax = plt.subplots()
ax.scatter(range(len(activity)), activity)
ax.set_xlabel('Electrode Number')
ax.set_ylabel('Mean Evoked Activity (z-score)')

# Finally, we'll mask for the times we want and concatenate across trials.
y = ecog._data[..., mask_time_ecog]
y = np.hstack(y).T
### y=output/DV encoding model ####
#we went from a 3D ecog._data.shape array (29,32,1500) to a 2D y.shape (23229,32)
#the mask time: -5 - 3.5s = 801 samples. 801 * 29 trials=23229 data points

#%% SPEECH ENVELOPE
env = mne.read_epochs(feature_path + 'envelope-epo.fif')

"""When building encoding or decoding models, it is important to account for the fact that neural activity generally isn't responding to stimuli
 at the exact same moment in time. It takes some time for a neural signal to make its way to the brain area under study. 
 Morever, it could be that a brain region cares about patterns of stimulus features as they change over time.
To account for these cases, we include time-lagged versions of each feature in the model fitting.
In this case, we'll use delays from .1 seconds before, to .4 seconds after the stimulus. 
Note that this can quickly make our feature space grow quite large, so be careful when you choose how many lags to use."""
# Define our time delays in seconds
delays = np.linspace(-0.4, 0, 20)
X_delayed_env = delay_time_series(env._data, delays, env.info['sfreq']) #this returns (n_epochs, n_features *n_channels*, n_delays, n_times)
#visualize the time-lagged features
fig, ax = plt.subplots()
ax.pcolormesh(env.times[mask_time_ecog], delays,
              X_delayed_env.mean(0)[0][..., mask_time_ecog], cmap='viridis')
ax.axis('tight')
ax.set_ylabel('Delay (s)')
ax.set_xlabel('Time (s)')
ax.set_title('Time-Delayed Speech Envelope')

#now combine this extended feature (X_delayed) and use it to fit our model
X_env = X_delayed_env.reshape([X_delayed_env.shape[0], -1, X_delayed_env.shape[-1]]) #this returns array of shape (29,20, 1500) (29 trials, 20 delayed time-points, 1500 sample points, no electrodes ofc)
X_env = X_env[..., mask_time_ecog]
X_env = np.hstack(X_env).T  #this returns 2D shape (23229,20) (again, -.5 to 3.5 =801 sample points, 23229)
#fit the model
est_env = make_pipeline(StandardScaler(), Ridge(alpha=1e5))
est_env.fit(X_env, y[:, ix_active]) #ix_active = electrode with most activity, electrode 32 (31 py count)
fig, ax = plt.subplots()
ax.plot(delays, est_env.steps[-1][-1].coef_) #for each delay, the coefs (n=20)
ax.set_xlabel('Time Delay')
ax.set_ylabel('Coefficient Value')

#%% SPECTROGRAM
spec = mne.read_epochs(feature_path + 'spectrogram-epo.fif') #(29, 64, 1501) 64 Frequencies
mask_time_spec = mne.utils._time_mask(spec.times, -.5, 3.5)
frequencies = np.array(spec.ch_names).astype(float)
X_delayed_spect = delay_time_series(spec._data, delays, spec.info['sfreq'])#(29,64,20,1501)
X_delayed_spect = X_delayed_spect.reshape(X_delayed_spect.shape[0], -1, len(spec.times))#this groups all delays together for all frequencies (=features, n=64). Result: (29,1280,1501)

# Model data
X_delayed_spect = np.hstack(X_delayed_spect[..., mask_time_spec]).T #returns 2D shape: (23229,1280) (20*64=1280 delay points)
# We'll use a higher ridge parameter since we have so many more coefficients
est_spect = make_pipeline(StandardScaler(), Ridge(alpha=1e7))
est_spect.fit(X_delayed_spect, y[:, ix_active])

# plot
coefs_spect = est_spect._final_estimator.coef_ #does the same trick as line 80. (1280), the 64 x 20 features
coefs_spect = coefs_spect.reshape(-1, len(delays)) #(64,20) 64 features, 20 time-delayed
f, ax = plt.subplots()
ax.pcolormesh(delays, frequencies, coefs_spect,
              cmap=plt.cm.coolwarm)
ax.set_xlabel('Time Delay (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Spectro-Temporal Receptive Field (STRF)')
ax.axis('tight')

#%% PHONEMES
phonemes = mne.read_epochs(feature_path + 'phonemes-epo.fif')
mask_time_ph = mne.utils._time_mask(phonemes.times, -.5, 3.5)
phonemes.drop_channels(['h#'])  # Drop silent times
X_delayed_ph = delay_time_series(phonemes._data, delays, phonemes.info['sfreq']) #(29,57,20,1501) #57 phonemes! 
X_delayed_ph = X_delayed_ph.reshape(X_delayed_ph.shape[0], -1, X_delayed_ph.shape[-1]) #(29,1140,1501) 57x20=1140
X_delayed_ph = np.hstack(X_delayed_ph[..., mask_time_ph]).T #shape(23229, 1140) 23229=801 sample points x 29 trials

# Now fit the model. We won't scale the inputs since they're binary
est_ph = Ridge(alpha=1e5)
est_ph.fit(X_delayed_ph, y)
coefs_ph = est_ph.coef_ #(32,1140) 32 channels, 1140 features (57 phonemes, 20 time-delayed features)
coefs_ph = coefs_ph.reshape(-1, len(phonemes.ch_names), len(delays))#(32,57,20)

###could also be done with make_pipeline. Same thing. You can comment this out###
est2 = make_pipeline( Ridge(alpha=1e5))
est2.fit(X_delayed_ph, y) #here not for channel 32 (31 py count) only. That way, we can see what channels are sensitive for phonemes and which are not 
coefs2=est2._final_estimator.coef_
coefs2 = coefs2.reshape(-1, len(phonemes.ch_names), len(delays))

#plot
f, ax = plt.subplots(figsize=(5, 15))
y_vals = np.arange(len(phonemes.ch_names))
ax.pcolormesh(delays, y_vals, coefs_ph[ix_active],
              cmap=plt.cm.RdBu_r)
plt.yticks(y_vals + .5, np.array(phonemes.ch_names)[y_vals])
ax.axis('tight')

#%% PHONETIC FEATURES
features=mne.read_epochs(feature_path + 'features-epo.fif')
mask_time_ph = mne.utils._time_mask(features.times, -.5, 3.5)
X_delayed_ft = delay_time_series(features._data, delays, features.info['sfreq']) #(29,8,20,1501) #8 phonetic features! 
X_delayed_ft = X_delayed_ft.reshape(X_delayed_ft.shape[0], -1, X_delayed_ft.shape[-1]) #(29,160,1501) 8x20=160
X_delayed_ft = np.hstack(X_delayed_ft[..., mask_time_ph]).T #shape(23229, 160) 23229=801 sample points x 29 trials

# Now fit the model. We won't scale the inputs since they're binary
est_ft = Ridge(alpha=1e5)
est_ft.fit(X_delayed_ft, y)
coefs_ft = est_ft.coef_ #(32,160) 32 channels, 160 features (8 phonetic features, 20 time-delayed features)
coefs_ft = coefs_ft.reshape(-1, len(features.ch_names), len(delays))#(32, 8,20)

#plot
f, ax = plt.subplots(figsize=(5, 15))
y_vals2 = np.arange(len(features.ch_names))
ax.pcolormesh(delays, y_vals2, coefs_ft[ix_active],
              cmap=plt.cm.RdBu_r)
plt.yticks(y_vals + .5, np.array(features.ch_names)[y_vals2])
ax.axis('tight')

#%% WORD ONSETS
onset = mne.read_epochs(feature_path + 'word_onset-epo.fif')
X_delayed_on = delay_time_series(onset._data, delays, env.info['sfreq']) #(29,1,20,1500)
X_delayed_on = X_delayed_on.reshape([X_delayed_on.shape[0], -1, X_delayed_on.shape[-1]]) #this returns array of shape (29,20, 1500) (29 trials, 20 delayed time-points, 1500 sample points, no electrodes ofc)
X_delayed_on = X_delayed_on[..., mask_time_ecog]
X_delayed_on = np.hstack(X_delayed_on).T  #this returns 2D shape (23229,20) (again, -.5 to 3.5 =801 sample points, 23229)

# Now fit the model. We won't scale the inputs since they're binary
est_on = Ridge(alpha=1e5)
est_on.fit(X_delayed_on, y)
coefs_on = est_on.coef_ #(32,20) 32 channels, 20 features (1 channel word onset, 20 time-delayed features)
coefs_on = coefs_on.reshape(-1, len(onset.ch_names), len(delays))#(32, 8,20)
fig, ax = plt.subplots()
onset_coefs=coefs_on[ix_active].reshape(20,)
ax.plot(delays, onset_coefs) #for each delay, the coefs (n=20)

#%% SHOWING ALL AT THE SAME TIME
fig, ax = plt.subplots(3, 1, sharex=True)

#envelope
ax[0].plot(delays, est_env.steps[-1][-1].coef_) #for each delay, the coefs (n=20)
ax[0].set_ylabel('Coefficient Value')
ax[0].set_title('Envelope')

#spectogram
ax[1].pcolormesh(delays, np.log10(frequencies), coefs_spect,
              cmap=plt.cm.coolwarm)
ax[1].set_ylabel('Frequency (Hz)')
ax[1].set_title('Spectro-Temporal Receptive Field (STRF)')
ax[1].axis('tight')

#phonemes
ax[2].pcolormesh(delays, y_vals, coefs_ph[ix_active],
              cmap=plt.cm.RdBu_r)
plt.yticks(y_vals + .5, np.array(phonemes.ch_names)[y_vals])
ax[2].axis('tight')
ax[2].set_title('Phonemes')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Phonemes')

#%% 
fig, ax = plt.subplots(2, 1, sharex=True)

#Phonetic Features
ax[0].pcolormesh(delays, y_vals2, coefs_ft[ix_active],
              cmap=plt.cm.RdBu_r)
plt.yticks(y_vals + .5, np.array(features.ch_names)[y_vals2])
ax[0].axis('tight')
ax[0].set_title('Phonetic Features')
ax[0].set_ylabel('Feature')

#Word onset
ax[1].plot(delays, onset_coefs) #for each delay, the coefs (n=20)
ax[1].set_title('Word Onset')
ax[1].set_ylabel('Coefficient Value')
ax[1].set_xlabel('Time (s)')







