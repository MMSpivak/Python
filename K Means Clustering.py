#Instead of saying this is a didget for cancer, we only say this is a didget
#K stands for how many clusters, attempts to divide data points into K classes
#move centroids to middle of their points and then reevaluate all the points and which centroid they are closest to
#do this over and over till no changes
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data) #.data part is all of the features so this scales it down to 0 and 1, saves time
y = digits.target

#k = len(np.unique(y)) dynamic way if changing data set
k = 10 #number of centroids
samples, features = data.shape #decomposes ??

def bench_k_means(estimator, name, data): #ways to score and make model more accurate
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

clf = KMeans(n_clusters= k, init= "random", n_init=10, )#TONS Of different parameters we can use, can find them online
bench_k_means(clf, "1", data)

