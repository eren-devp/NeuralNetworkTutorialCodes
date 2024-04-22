import sklearn
import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.cluster import KMeans

# https://www.youtube.com/watch?v=g1Zbuk1gAfk&list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr&index=11
# I didn't understand it much T-T

digits = load_digits()
data = scale(digits.data)  # Scaling out data features between -1 and +1.
y = digits.target
k = len(np.unique(y))  # Digit length.
samples, features = data.shape

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, y, test_size=0.2)

def bench_k_means(estimator, name, data):  # Visualizing function, just copied lol.
    """
    :param estimator The classifier.
    :param name Name of the bench.
    :param data The data to show.
    """
    # https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
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


clf = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300)  # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
bench_k_means(clf, "1", data)
