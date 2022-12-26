from util import trainer_util, metrics
import numpy as np
labels = np.array([0,1,3,1,0,0,0,0,1,255,36,1,1,1,1,1,0,0,0,1,1,1,255,255,255,255])
pred = np.array([0.25, 0.89, 0.03, 0.02, 0.8, 0.9, 0.37, 0.001, 0.89, 0.33,0.89,0.59, 0.44, 0.78, 0.96, 0.23, 0.56, 0.98, 0.26, 0.75, 0.69, 0.96, 0.56, 0.31, 0.13, 0.78])
results = metrics.get_metrics(labels, pred)
print("ok")