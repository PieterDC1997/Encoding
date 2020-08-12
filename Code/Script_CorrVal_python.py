####################
###IMPORT MODULES###
####################
import os, collections
import os.path as op
import mne
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from modelingtools import (delay_time_series, plot_cv_indices,
                           cross_validate_alpha)
import modelingtools
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import scale, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from itertools import product
from scipy import stats
import warnings
#%% SET FILE PATHS
os.getcwd()
os.chdir('C/.../Tutorial')
cwd='C/.../Tutorial'
raw_path='C/.../Tutorial/raw_data/'
feature_path='C/.../Tutorial/features/'
sfreq_new = 100  # Cut sfreq in half to save computation time
tmin, tmax = -.5, 4
#%% ECOG DATA
ecog = mne.read_epochs(feature_path + 'hfa_ecog-epo.fif', preload=True)
ecog.resample(100)

#how many trials
trials=np.arange(len(ecog))
trials=len(trials)

#channels
channels=32

#code to select electrode with highest response to speech stimuli 
activity = ecog.copy()
_ = mne.baseline.rescale(activity._data, activity.times, (None, 0),
                         mode='zscore', copy=False)
activity = activity._data.mean(0).mean(-1)
use_elec = np.argmax(activity) #electrode 32 again (31 py count)
#%% SPECTROGRAM
"""Read end comments of this cell. So, of you'd like, you can skip this cell. We'll repeat the spectro anyway"""
#our ridge regression model
model = Ridge(alpha=1e5)

# Read in the data
spec = mne.read_epochs(feature_path + 'spectrogram-epo.fif')
# Take the log so it's more normally-distributed in each frequency band
spec._data = np.log(spec._data)
#resample
spec.resample(sfreq_new)

# Times to use in fitting
mask_time = mne.utils._time_mask(ecog.times, -.2, 4)
# We'll create the delays for the spectrogram ahead of time
delays = np.linspace(0, 0.4, 20)
X_delayed = delay_time_series(spec._data, delays, spec.info['sfreq'])
X = X_delayed.reshape(X_delayed.shape[0], -1, X_delayed.shape[-1])
X = X[..., mask_time]

#We'll create y with ALL electrodes, and see where correlations are highest
y = ecog._data[..., mask_time]

# setup KFold, n_splits
kf=KFold(n_splits=5, shuffle=True)

scores = []
pearsons = []
spearmans = []
for tr, tt in kf.split(X):
    # Pull the training / testing data for the ecog data
    print(tr)
    print(tt)
    y_tr = np.hstack(y[tr]).T    
    y_tt = np.hstack(y[tt]).T

    # Pull the training / testing data for the spectrogram
    X_tr = np.hstack(X[tr]).T
    X_tt = np.hstack(X[tt]).T
    
    # Scale all the features for simplicity
    X_tr = scale(X_tr)
    X_tt = scale(X_tt)
    y_tr = scale(y_tr)
    y_tt = scale(y_tt)
    
    # Fit the model, and use it to predict on new data
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_tt)
    print(r2_score(y_tt, predictions))
    
    # print the coefficient of determination (R2)
    for y_true, y_pred in zip(y_tt.T, predictions.T):
        scores.append(r2_score(y_true, y_pred))
        spearman=stats.spearmanr(y_true, y_pred)
        pearson=stats.pearsonr(y_true, y_pred)
        spearmans.append(spearman[0])
        pearsons.append(pearson[0])
    f, ax= plt.subplots()
    ax.plot(scale(y_tt), color='k', alpha=.2, lw=2)
    ax.plot(scale(predictions), color='r', lw=2)

av_scores=[]
av_pearsons=[]
av_spearmans=[]
for i in range(32):
    av_scores.append(np.average(scores[i::32]))
    av_pearsons.append(np.average(pearsons[i::32]))
    av_spearmans.append(np.average(spearmans[i::32]))
im = plt.imread(raw_path + 'brain.png')
melec = pd.read_csv(raw_path + 'meta_elec.csv')
xy = melec[['x_2d', 'y_2d']].values
#plot R2 scores
ax = modelingtools.plot_activity_on_brain(av_scores, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Prediction Scores ($R^2$)', fontsize=20)
#plot pearsons
ax = modelingtools.plot_activity_on_brain(av_pearsons, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Pearsons R', fontsize=20)
#plot spearmans
ax = modelingtools.plot_activity_on_brain(av_spearmans, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Spearmans R', fontsize=20)

max_elec=np.argmax(av_scores) #with this line, with the figures and by printing the av lists, we can see that electrode 12 (11 np count)
                              #has the highest determination coefficient. Now, if you want, for better visualization purposes,
                              #you can adapt y to y=X[:, max_elec, :][..., mask_time] and rerun this cell.
                              #then, the figures in the for loop (to split the X matrix) will only include a single electrode
                              #also comment out lines in that for loop where you append the scores, spearmans and pearsons


### NOW, WE CAN USE CV TO FIT HYPERPARAMETER ALPHA ###

values=np.logspace(1,8,8)
model=RidgeCV(alphas=values, cv=5)
y = ecog._data[..., mask_time]
for tr, tt in kf.split(X):
    print(tr)
    print(tt)
    # Pull the training / testing data for the ecog data
    y_tr = np.hstack(y[tr]).T    
    y_tt = np.hstack(y[tt]).T

    # Pull the training / testing data for the spectrogram
    X_tr = np.hstack(X[tr]).T
    X_tt = np.hstack(X[tt]).T
    
    # Scale all the features for simplicity
    X_tr = scale(X_tr)
    X_tt = scale(X_tt)
    y_tr = scale(y_tr)
    y_tt = scale(y_tt)
    
    # Fit the model, and use it to predict on new data
    model.fit(X_tr, y_tr)
    predictions=model.predict(X_tt)
    print(model.alpha_)
    print(r2_score(y_tt, predictions))
    
"""Now, let's standardize our steps and make it parallel for all features

I have mixed some steps together. This is bcs I first explained CV with the standard method we were applying (i.e. with a fixed alpha)
We should never use a fixed alpha, but rather let the CV procedure take it for us.

This is the standard procedure now:
    1. Create the time-delayed features
    2. Set up X-matrix
    3. Set up y-matrix, with ALL electrodes!
    4. Do the CV, RidgeCV with alphas 1e0 - 1e8. Model automatically selects the best alpha
    5. Fit the CV model
    6. Predict the EEG signal
    7. compute pearson's r, Spearman's r and R-squared values with the actual EEG signal
    8. You can then just look at the electrode with highest correlations/R2 values and plot this for visualization purposes if necessary"""

#%% ENVELOPE
#our ridge regression model
alpha_values=np.logspace(1,8,8)
model=RidgeCV(alphas=alpha_values, cv=5)

# Times to use in fitting
mask_time = mne.utils._time_mask(ecog.times, -.2, 4)

###y, ALL electrodes###
y = ecog._data[..., mask_time]

###create X_env###
env = mne.read_epochs(feature_path + 'envelope-epo.fif')
#resample
env.resample(sfreq_new)
#delays
delays = np.linspace(0, 0.4, 20)
#X
X_env = delay_time_series(env._data, delays, env.info['sfreq'])
X_env = X_env.reshape(X_env.shape[0], -1, X_env.shape[-1])
X_env = X_env[..., mask_time]

###setup KFold, n_splits###
kf=KFold(n_splits=5, shuffle=True)

###now setup the loop to iterate over KFold. This is the CV, with k=5
scores = []
pearsons = []
spearmans = []
for tr_env, tt_env in kf.split(X_env):
    # Pull the training / testing data for the ecog data
    y_tr_env = np.hstack(y[tr_env]).T    
    y_tt_env = np.hstack(y[tt_env]).T
    # Pull the training / testing data
    X_tr_env = np.hstack(X_env[tr_env]).T
    X_tt_env = np.hstack(X_env[tt_env]).T
    # Scale all the features for simplicity
    X_tr_env = scale(X_tr_env)
    X_tt_env = scale(X_tt_env)
    y_tr_env = scale(y_tr_env)
    y_tt_env = scale(y_tt_env)
    
    ###Fit the model, and use it to predict on new data###
    model.fit(X_tr_env, y_tr_env)
    predictions_env = model.predict(X_tt_env)
    print(r2_score(y_tt_env, predictions_env)) #prints the env data
    
    # print the coefficient of determination (R2)
    for y_true, y_pred in zip(y_tt_env.T, predictions_env.T):
        scores.append(r2_score(y_true, y_pred))
        spearman=stats.spearmanr(y_true, y_pred)
        pearson=stats.pearsonr(y_true, y_pred)
        spearmans.append(spearman[0])
        pearsons.append(pearson[0])
    f, ax= plt.subplots()
    ax.plot(scale(y_tt_env), color='k', alpha=.2, lw=2)
    ax.plot(scale(predictions_env), color='r', lw=2)
    print(model.alpha_) #print which alpha the RidgeCV chose

###Get average of all 32nd value, so we have an average per electrode over the k (5) folds### 
av_scores=[]
av_pearsons=[]
av_spearmans=[]
for i in range(channels):
    av_scores.append(np.average(scores[i::channels]))
    av_pearsons.append(np.average(pearsons[i::channels]))
    av_spearmans.append(np.average(spearmans[i::channels]))

#Plot these values on the brain
im = plt.imread(raw_path + 'brain.png')
melec = pd.read_csv(raw_path + 'meta_elec.csv')
xy = melec[['x_2d', 'y_2d']].values
#plot R2 scores
ax = modelingtools.plot_activity_on_brain(av_scores, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Prediction Scores ($R^2$)', fontsize=20)
#plot pearsons
ax = modelingtools.plot_activity_on_brain(av_pearsons, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Pearsons R', fontsize=20)
#plot spearmans
ax = modelingtools.plot_activity_on_brain(av_spearmans, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Spearmans R', fontsize=20)

#electrode with highest values
max_elec_scores=np.argmax(av_scores)
max_elec_p=np.argmax(av_pearsons)
max_elec_s=np.argmax(av_spearmans)
print(max_elec_scores, max_elec_p, max_elec_s)  #with this line, with the figures and by printing the av lists, we can see that electrode 5 (4 np count)
                                                #has the highest determination coefficient. Now, if you want, for better visualization purposes,
                                                #you can adapt y to y=X[:, max_elec, :][..., mask_time] and rerun this cell.
                                                #then, the figures in the for loop (to split the X matrix) will only include a single electrode
                                                #also comment out lines in that for loop where you append the scores, spearmans and pearsons, otherwise it will get stuck
                                                #note that electrode 12 (11) has the highest spearman R

#%% SPECTROGRAM
#our ridge regression model
alpha_values=np.logspace(1,8,8)
model=RidgeCV(alphas=alpha_values, cv=5)

# Times to use in fitting
mask_time = mne.utils._time_mask(ecog.times, -.2, 4)

###y, ALL electrodes###
y = ecog._data[..., mask_time]

###create X_spec###
spec = mne.read_epochs(feature_path + 'spectrogram-epo.fif')
# Take the log so it's more normally-distributed in each frequency band
spec._data = np.log(spec._data)
#resample
spec.resample(sfreq_new)
#delays
delays = np.linspace(0, 0.4, 20)
#X
X_spec = delay_time_series(spec._data, delays, spec.info['sfreq'])
X_spec = X_spec.reshape(X_spec.shape[0], -1, X_spec.shape[-1])
X_spec = X_spec[..., mask_time]

###setup KFold, n_splits###
kf=KFold(n_splits=5, shuffle=True)

###now setup the loop to iterate over KFold. This is the CV, with k=5
scores = []
pearsons = []
spearmans = []
for tr_spec, tt_spec in kf.split(X_spec):
    # Pull the training / testing data for the ecog data
    y_tr_spec = np.hstack(y[tr_spec]).T    
    y_tt_spec = np.hstack(y[tt_spec]).T
    # Pull the training / testing data
    X_tr_spec = np.hstack(X_spec[tr_spec]).T
    X_tt_spec = np.hstack(X_spec[tt_spec]).T
    # Scale all the features for simplicity
    X_tr_spec = scale(X_tr_spec)
    X_tt_spec = scale(X_tt_spec)
    y_tr_spec = scale(y_tr_spec)
    y_tt_spec = scale(y_tt_spec)
    
    ###Fit the model, and use it to predict on new data###
    model.fit(X_tr_spec, y_tr_spec)
    predictions_spec = model.predict(X_tt_spec)
    print(r2_score(y_tt_spec, predictions_spec)) #prints the env data
    
    # print the coefficient of determination (R2)
    for y_true, y_pred in zip(y_tt_spec.T, predictions_spec.T):
        scores.append(r2_score(y_true, y_pred))
        spearman=stats.spearmanr(y_true, y_pred)
        pearson=stats.pearsonr(y_true, y_pred)
        spearmans.append(spearman[0])
        pearsons.append(pearson[0])
    f, ax= plt.subplots()
    ax.plot(scale(y_tt_spec), color='k', alpha=.2, lw=2)
    ax.plot(scale(predictions_spec), color='r', lw=2)
    print(model.alpha_) #print which alpha the RidgeCV chose

###Get average of all 32nd value, so we have an average per electrode over the k (5) folds### 
av_scores=[]
av_pearsons=[]
av_spearmans=[]
for i in range(channels):
    av_scores.append(np.average(scores[i::channels]))
    av_pearsons.append(np.average(pearsons[i::channels]))
    av_spearmans.append(np.average(spearmans[i::channels]))

#Plot these values on the brain
im = plt.imread(raw_path + 'brain.png')
melec = pd.read_csv(raw_path + 'meta_elec.csv')
xy = melec[['x_2d', 'y_2d']].values
#plot R2 scores
ax = modelingtools.plot_activity_on_brain(av_scores, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Prediction Scores ($R^2$)', fontsize=20)
#plot pearsons
ax = modelingtools.plot_activity_on_brain(av_pearsons, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Pearsons R', fontsize=20)
#plot spearmans
ax = modelingtools.plot_activity_on_brain(av_spearmans, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Spearmans R', fontsize=20)

#electrode with highest values
max_elec_scores=np.argmax(av_scores)
max_elec_p=np.argmax(av_pearsons)
max_elec_s=np.argmax(av_spearmans)
print(max_elec_scores, max_elec_p, max_elec_s) #electrode 12 (11py). Again, you can adapt your code for visualization purposes

#%% PHONEMES
#our ridge regression model
alpha_values=np.logspace(1,8,8)
model=RidgeCV(alphas=alpha_values, cv=5)

# Times to use in fitting
mask_time = mne.utils._time_mask(ecog.times, -.2, 4)

###y, ALL electrodes###
y = ecog._data[..., mask_time]

###create X_pns###
pns = mne.read_epochs(feature_path + 'phonemes-epo.fif')
#resample
pns.resample(sfreq_new)
#delays
delays = np.linspace(0, 0.4, 20)
#X
X_pns = delay_time_series(pns._data, delays, pns.info['sfreq'])
X_pns = X_pns.reshape(X_pns.shape[0], -1, X_pns.shape[-1])
X_pns = X_pns[..., mask_time]

###setup KFold, n_splits###
kf=KFold(n_splits=5, shuffle=True)

###now setup the loop to iterate over KFold. This is the CV, with k=5
scores = []
pearsons = []
spearmans = []
for tr_pns, tt_pns in kf.split(X_pns):
    # Pull the training / testing data for the ecog data
    y_tr_pns = np.hstack(y[tr_pns]).T    
    y_tt_pns = np.hstack(y[tt_pns]).T
    # Pull the training / testing data
    X_tr_pns = np.hstack(X_pns[tr_pns]).T
    X_tt_pns = np.hstack(X_pns[tt_pns]).T
    # We won't scale since our values are binary
    
    ###Fit the model, and use it to predict on new data###
    model.fit(X_tr_pns, y_tr_pns)
    predictions_pns = model.predict(X_tt_pns)
    print(r2_score(y_tt_pns, predictions_pns)) #prints the env data
    
    # print the coefficient of determination (R2)
    for y_true, y_pred in zip(y_tt_pns.T, predictions_pns.T):
        scores.append(r2_score(y_true, y_pred))
        spearman=stats.spearmanr(y_true, y_pred)
        pearson=stats.pearsonr(y_true, y_pred)
        spearmans.append(spearman[0])
        pearsons.append(pearson[0])
    f, ax= plt.subplots()
    ax.plot(scale(y_tt_pns), color='k', alpha=.2, lw=2)
    ax.plot(scale(predictions_pns), color='r', lw=2)
    print(model.alpha_) #print which alpha the RidgeCV chose

###Get average of all 32nd value, so we have an average per electrode over the k (5) folds### 
av_scores=[]
av_pearsons=[]
av_spearmans=[]
for i in range(channels):
    av_scores.append(np.average(scores[i::channels]))
    av_pearsons.append(np.average(pearsons[i::channels]))
    av_spearmans.append(np.average(spearmans[i::channels]))

#Plot these values on the brain
im = plt.imread(raw_path + 'brain.png')
melec = pd.read_csv(raw_path + 'meta_elec.csv')
xy = melec[['x_2d', 'y_2d']].values
#plot R2 scores
ax = modelingtools.plot_activity_on_brain(av_scores, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Prediction Scores ($R^2$)', fontsize=20)
#plot pearsons
ax = modelingtools.plot_activity_on_brain(av_pearsons, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Pearsons R', fontsize=20)
#plot spearmans
ax = modelingtools.plot_activity_on_brain(av_spearmans, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Spearmans R', fontsize=20)

#electrode with highest values
max_elec_scores=np.argmax(av_scores)
max_elec_p=np.argmax(av_pearsons)
max_elec_s=np.argmax(av_spearmans)
print(max_elec_scores, max_elec_p, max_elec_s) #electrode 12 (11py). Again, you can adapt your code for visualization purposes

#%% PHONETIC FEATURES
#our ridge regression model
alpha_values=np.logspace(1,8,8)
model=RidgeCV(alphas=alpha_values, cv=5)

# Times to use in fitting
mask_time = mne.utils._time_mask(ecog.times, -.2, 4)

###y, ALL electrodes###
y = ecog._data[..., mask_time]

###create X_ft###
ft = mne.read_epochs(feature_path + 'features-epo.fif')
#resample
ft.resample(sfreq_new)
#delays
delays = np.linspace(0, 0.4, 20)
#X
X_ft = delay_time_series(ft._data, delays, ft.info['sfreq'])
X_ft = X_ft.reshape(X_ft.shape[0], -1, X_ft.shape[-1])
X_ft = X_ft[..., mask_time]

###setup KFold, n_splits###
kf=KFold(n_splits=5, shuffle=True)

###now setup the loop to iterate over KFold. This is the CV, with k=5
scores = []
pearsons = []
spearmans = []
for tr_ft, tt_ft in kf.split(X_ft):
    # Pull the training / testing data for the ecog data
    y_tr_ft = np.hstack(y[tr_ft]).T    
    y_tt_ft = np.hstack(y[tt_ft]).T
    # Pull the training / testing data
    X_tr_ft = np.hstack(X_ft[tr_ft]).T
    X_tt_ft = np.hstack(X_ft[tt_ft]).T
    # We won't scale since our values are binary
    
    ###Fit the model, and use it to predict on new data###
    model.fit(X_tr_ft, y_tr_ft)
    predictions_ft = model.predict(X_tt_ft)
    print(r2_score(y_tt_ft, predictions_ft)) #prints the env data
    
    # print the coefficient of determination (R2)
    for y_true, y_pred in zip(y_tt_ft.T, predictions_ft.T):
        scores.append(r2_score(y_true, y_pred))
        spearman=stats.spearmanr(y_true, y_pred)
        pearson=stats.pearsonr(y_true, y_pred)
        spearmans.append(spearman[0])
        pearsons.append(pearson[0])
    f, ax= plt.subplots()
    ax.plot(scale(y_tt_ft), color='k', alpha=.2, lw=2)
    ax.plot(scale(predictions_ft), color='r', lw=2)
    print(model.alpha_) #print which alpha the RidgeCV chose

###Get average of all 32nd value, so we have an average per electrode over the k (5) folds### 
av_scores=[]
av_pearsons=[]
av_spearmans=[]
for i in range(channels):
    av_scores.append(np.average(scores[i::channels]))
    av_pearsons.append(np.average(pearsons[i::channels]))
    av_spearmans.append(np.average(spearmans[i::channels]))

#Plot these values on the brain
im = plt.imread(raw_path + 'brain.png')
melec = pd.read_csv(raw_path + 'meta_elec.csv')
xy = melec[['x_2d', 'y_2d']].values
#plot R2 scores
ax = modelingtools.plot_activity_on_brain(av_scores, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Prediction Scores ($R^2$)', fontsize=20)
#plot pearsons
ax = modelingtools.plot_activity_on_brain(av_pearsons, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Pearsons R', fontsize=20)
#plot spearmans
ax = modelingtools.plot_activity_on_brain(av_spearmans, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Spearmans R', fontsize=20)

#electrode with highest values
max_elec_scores=np.argmax(av_scores)
max_elec_p=np.argmax(av_pearsons)
max_elec_s=np.argmax(av_spearmans)
print(max_elec_scores, max_elec_p, max_elec_s) #electrode 12 (11py). Again, you can adapt your code for visualization purposes

#%% WORD ONSETS
#our ridge regression model
alpha_values=np.logspace(1,8,8)
model=RidgeCV(alphas=alpha_values, cv=5)

# Times to use in fitting
mask_time = mne.utils._time_mask(ecog.times, -.2, 4)

###y, ALL electrodes###
y = ecog._data[..., mask_time]

###create X_wo###
wo = mne.read_epochs(feature_path + 'word_onset-epo.fif')
#resample
wo.resample(sfreq_new)
#delays
delays = np.linspace(0, 0.4, 20)
#X
X_wo = delay_time_series(wo._data, delays, wo.info['sfreq'])
X_wo = X_wo.reshape(X_wo.shape[0], -1, X_wo.shape[-1])
X_wo = X_wo[..., mask_time]

###setup KFold, n_splits###
kf=KFold(n_splits=5, shuffle=True)

###now setup the loop to iterate over KFold. This is the CV, with k=5
scores = []
pearsons = []
spearmans = []
for tr_wo, tt_wo in kf.split(X_wo):
    # Pull the training / testing data for the ecog data
    y_tr_wo = np.hstack(y[tr_wo]).T    
    y_tt_wo = np.hstack(y[tt_wo]).T
    # Pull the training / testing data
    X_tr_wo = np.hstack(X_wo[tr_wo]).T
    X_tt_wo = np.hstack(X_wo[tt_wo]).T
    # We won't scale since our values are binary
    
    ###Fit the model, and use it to predict on new data###
    model.fit(X_tr_wo, y_tr_wo)
    predictions_wo = model.predict(X_tt_wo)
    print(r2_score(y_tt_wo, predictions_wo)) #prints the env data
    
    # print the coefficient of determination (R2)
    for y_true, y_pred in zip(y_tt_wo.T, predictions_wo.T):
        scores.append(r2_score(y_true, y_pred))
        spearman=stats.spearmanr(y_true, y_pred)
        pearson=stats.pearsonr(y_true, y_pred)
        spearmans.append(spearman[0])
        pearsons.append(pearson[0])
    f, ax= plt.subplots()
    ax.plot(scale(y_tt_wo), color='k', alpha=.2, lw=2)
    ax.plot(scale(predictions_wo), color='r', lw=2)
    print(model.alpha_) #print which alpha the RidgeCV chose

###Get average of all 32nd value, so we have an average per electrode over the k (5) folds### 
av_scores=[]
av_pearsons=[]
av_spearmans=[]
for i in range(channels):
    av_scores.append(np.average(scores[i::channels]))
    av_pearsons.append(np.average(pearsons[i::channels]))
    av_spearmans.append(np.average(spearmans[i::channels]))

#Plot these values on the brain
im = plt.imread(raw_path + 'brain.png')
melec = pd.read_csv(raw_path + 'meta_elec.csv')
xy = melec[['x_2d', 'y_2d']].values
#plot R2 scores
ax = modelingtools.plot_activity_on_brain(av_scores, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Prediction Scores ($R^2$)', fontsize=20)
#plot pearsons
ax = modelingtools.plot_activity_on_brain(av_pearsons, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Pearsons R', fontsize=20)
#plot spearmans
ax = modelingtools.plot_activity_on_brain(av_spearmans, im, xy[:, 0], xy[:, 1],
                                      size_scale=3000, vmin=-.1, vmax=.1,
                                      cmap=modelingtools.cmap_score)
ax.figure.set_size_inches(10, 10)
ax.set_title('Spearmans R', fontsize=20)
#electrode with highest values
max_elec_scores=np.argmax(av_scores)
max_elec_p=np.argmax(av_pearsons)
max_elec_s=np.argmax(av_spearmans)
print(max_elec_scores, max_elec_p, max_elec_s) #electrodes differ... Spearman does however keeps electrode 11 as highest. Again, you can adapt your code for visualization purposes
