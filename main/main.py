# executable python file

import numpy as np  # linear algebra
import pandas as pd  # data handling
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # ^
import plotly as py  # ^^
import plotly.graph_objs as go  # ^^^
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder




# read the dataset

data = pd.read_csv("online_shoppers_intention.csv")

#bounce rates of users

x = data.iloc[:, [5, 6]].values
x.shape

#number of clustering groups 

wcss = []      #within cluster sum of squares
for i in range(1, 11):
    km = KMeans(n_clusters = i,  #using kmeans to find 1 to 11 clusters, saving wcss for each number
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'full',
              tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)

#plotting wcss vs number of clusters
plt.rcParams['figure.figsize'] = (9.28, 5)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

n = 2 #n is the number of optimal clustering groups found using elbow method above, change as necessary

km = KMeans(n_clusters = n, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
# get predicted cluster index for each sample: 0, 1, 2
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 50, c = 'yellow', label = 'Uninterested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 50, c = 'pink', label = 'Interested Customers')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

plt.title('ProductRelated Duration vs Bounce Rate', fontsize = 20)
plt.grid()
plt.xlabel('ProductRelated Duration')
plt.ylabel('Bounce Rates') 
plt.legend()
plt.show()

le = LabelEncoder()
labels_true = le.fit_transform(data['Revenue'])

# get predicted clustering result label
labels_pred = y_means

# print adjusted rand index, which measures the similarity of the two assignments
from sklearn import metrics
score = metrics.adjusted_rand_score(labels_true, labels_pred)
print("Adjusted rand index: ")
print(score)

# print confusion matrix
#cm = metrics.plot_confusion_matrix(None, labels_true, labels_pred)
#print(cm)

import scikitplot as skplt
plt_1 = skplt.metrics.plot_confusion_matrix(labels_true, labels_pred, normalize=False)
plt_2 = skplt.metrics.plot_confusion_matrix(labels_true, labels_pred, normalize=True)
plt.show()