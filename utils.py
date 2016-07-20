"""
Utils functions.
"""

import numpy as np
from glob import glob
import re
import pandas as pd
import seaborn as sns
from pylab import plt

from pyriemann.utils.covariance import _lwf
from numpy import ones, kron, mean, eye, hstack, dot, tile
from scipy.linalg import pinv

from sklearn.metrics import confusion_matrix

ix_center = [1, 2, 3, 20]
ix_left = [4, 5, 6, 7, 21, 22]
ix_right = [8, 9, 10, 11, 23, 24]

ix_tot = ix_center + ix_left + ix_right

Task_names = ["Forearm to table", "Forearm to Box", "Extend Elbow 1",
              "Extend Elbow 2", "Hand to the Table", "Hand to the Box",
              "Reach and Retrieve", "Lift a can", "Lift a Pencil",
              "Lift a Paper Clip", "Stack Checkers", "Flip Cards",
              "Turn Key", "Fold Towel", "Lift Baskets"]


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                          cmap=plt.cm.Blues, ax=None):
    """Plot Confusion Matrix."""
    df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
    sns.heatmap(df, annot=True, fmt=".1f", linewidths=.5, vmin=0, vmax=100,
                cmap=cmap, ax=ax, cbar=False)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def generate_report(pred_tot, labels, name='SVM'):
    """Generate and plot accuracy report."""
    fig, axes = plt.subplots(2, 2, figsize=[10, 10])
    pred_tot = np.array(pred_tot)

    acc = []
    for preds in pred_tot:
        preds = np.argmax(preds, axis=1)
        acc.append(100*np.mean(preds == labels))

    res = pd.DataFrame(data=acc, columns=['acc'], index=Task_names)
    res['task'] = Task_names
    res.to_csv('./results/results_individual_tasks_%s.csv' % name)

    g = sns.barplot(x='acc', y='task', data=res.sort('acc', ascending=False),
                    palette="Blues", orient='h', ax=axes[0, 0])
    g.set_xlim([30, 100])
    g.set_xlabel('Accuracy (%)')
    g.set_title('Accuracy per task (chance level: 33%)')
    g.set_ylabel('')

    acc = []
    ix = np.argsort(res.acc)[::-1].values
    for i in range(15):
        preds = np.argmax(np.mean((pred_tot[ix[:i+1]]), axis=0), axis=1)
        acc.append(np.mean(preds == labels))

    n_opt = np.argmax(acc) + 1
    res = pd.DataFrame(data=acc, columns=['acc'], index=range(1, 16))
    res.to_csv('./results/results_cumul_%s.csv' % name)

    g = sns.tsplot(acc, range(1, 16), ax=axes[1, 0])
    axes[1, 0].plot([n_opt, n_opt], [0.8, 0.95], ls='--', lw=2, c='r')
    axes[1, 0].set_ylim(0.8, 0.95)
    g.set_xlabel('Number of task')
    g.set_ylabel('Accuracy (%)')

    preds = np.argmax(np.mean(pred_tot[ix[0:n_opt]], axis=0), axis=1)

    acc = np.mean(preds == labels)
    tm = ['Healthy', 'Mild', 'Moderate']

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    np.set_printoptions(precision=2)
    cm_normalized = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    title = 'Accuracy : %.1f percent' % (acc*100)
    plot_confusion_matrix(cm_normalized, tm, title=title, ax=axes[1, 1])

    c_alpha = CronbachAlpha([labels, preds])
    Icc, _, _, _, _, _ = ICC_rep_anova(np.array([labels, preds]).T)

    axes[0, 1].text(0.2, 0.8, 'Cronbach alpha : %.3f' % c_alpha)
    axes[0, 1].text(0.2, 0.7, 'Interclass Corr : %.3f' % Icc)

    return fig, axes


def read_data(subject_names,
              estimator=_lwf, Ntasks=15, Base='./data',
              centroid=ix_tot):
    """Read data."""

    ix_full = np.concatenate([[3*i, 3*i+1, 3*i+2] for i in centroid])
    condition_names = ['healthy', 'mild', 'moderate']
    X = []
    subject = []
    condition = []
    task = []
    timing = []

    reg = re.compile('.*/(.*)_(.*)_task(\d*).bin')

    for name in subject_names:

        for c in condition_names:
            invalid = False
            fnames = []
            for t in range(1, Ntasks + 1):
                fi = glob('%s/%s_%s_task%02d.bin' % (Base, name, c, t))
                if len(fi) > 0:
                    fnames.append(fi[0])
                else:
                    print("can't find cond. %s task %d for subject %s" %
                          (c, t, name))
                    invalid = True
            if invalid:
                print('skip subject %s' % name)
                continue

            for fname in fnames:
                # read binary file
                data = np.fromfile(fname, np.float32)

                # reshape binary file
                data = data.reshape((len(data)/75, 75)).T

                if data.shape[1] > 0:
                    # estimate cov matrix
                    tmp = 1e3*data[ix_full, :]
                    Nc, Ns = tmp.shape
                    X.append(estimator(tmp))
                    timing.append(data.shape[1])

                    # regexp to find the subject
                    s, c, t = reg.findall(fname)[0]
                    subject.append(s)
                    condition.append(c)
                    task.append(int(t))
                else:
                    print('Empty file for %s' % fname)

    # convert python list into array
    X = np.array(X)
    subject = np.array(subject)
    condition = np.array(condition)
    task = np.array(task)
    timing = np.array(timing)
    return X, subject, condition, task, timing


def CronbachAlpha(itemscores):
    """Estimates the CrombachAlpha."""
    itemscores = np.asarray(itemscores).T
    itemvars = itemscores.var(axis=0, ddof=1)
    tscores = itemscores.sum(axis=1)
    nitems = itemscores.shape[1]
    calpha = (nitems / float(nitems-1) *
              (1 - itemvars.sum() / float(tscores.var(ddof=1))))
    return calpha


def ICC_rep_anova(Y):
    '''
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns
    One Sample Repeated measure ANOVA
    Y = XB + E with X = [FaTor / Subjects]
    '''

    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = kron(eye(nb_conditions), ones((nb_subjects, 1)))  # sessions
    x0 = tile(eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = hstack([x, x0])

    # Sum Square Error
    predicted_Y = dot(dot(dot(X, pinv(dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((mean(Y, 0) - mean_Y) ** 2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) /
    # (mean square subjeT + (k-1)*-mean square error)
    ICC = (MSR - MSE) / (MSR + dfc * MSE)

    e_var = MSE# variance of error
    r_var = (MSR - MSE)/nb_conditions# variance between subjects

    return ICC, r_var, e_var, session_effect_F, dfc, dfe
