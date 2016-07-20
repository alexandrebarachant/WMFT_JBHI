"""
Classification per condition with probability average across tasks.
"""
import numpy as np

from pylab import plt

from pyriemann.tangentspace import TangentSpace

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.svm import SVC

from pyriemann.utils.covariance import _lwf

from utils import read_data, generate_report


subject_names = ['S%02d' % i for i in range(1, 25)]

Base = './data'
Nconditions = 3
# read data


X, subject, condition, task, timing = read_data(subject_names, Base=Base,
                                                estimator=_lwf)

clf = make_pipeline(TangentSpace('logeuclid'),
                    StandardScaler(with_mean=False),
                    SVC(C=100, kernel='linear',
                        probability=True, decision_function_shape='ovo',
                        random_state=454111))

# initialize variables
acc = []
pred_tot = []
acc_tot = []
labels_tot = []

# Encore labels (from category to numeric value)
encoder = LabelEncoder()
labels = encoder.fit_transform(condition)

# Loop over each task
for ta in np.unique(task):

    # Find index of data corresponding of the current task
    ix_task = task == ta

    # restrict data to the current task
    labels2 = labels[ix_task]
    X2 = X[ix_task]

    # initialize empty array to store prediction
    preds = np.zeros((labels2.shape[0], Nconditions))

    # create cross validator
    cv = LeaveOneLabelOut(subject[ix_task])
    acc = []

    # do cross validation
    for train, test in cv:
        clf.fit(X2[train], labels2[train])
        preds[test] += clf.predict_proba(X2[test])

    # store results
    print np.mean(np.argmax(preds, axis=1) == labels2)
    pred_tot.append(preds)
    labels_tot.append(labels2)

# average prediction across all tasks
preds = np.argmax(np.mean(pred_tot, axis=0), axis=1)
labels = np.mean(labels_tot, axis=0)
pred_tot = np.array(pred_tot)

generate_report(pred_tot, labels, name='SVM')
plt.savefig('./results/results_svm.png')
plt.show()
