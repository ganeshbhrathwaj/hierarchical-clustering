import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
ds=pd.read_csv('Mall_Customers.csv')
x=ds.iloc[:,[3,4]].values

#using dendrogram to no of clustures
import scipy.cluster.hierarchy as sch
d=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('dendrogram')
plt.xlabel('customer')
plt.ylabel('euclidien dist')
plt.show()

#fitting hierarchical clustring 
from sklearn.cluster import  AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
yhc=hc.fit_predict(x)

#visualizing the data
plt.scatter(x[yhc==0,0],x[yhc==0,1],s=100,c='red',label='cluster 1')
plt.scatter(x[yhc==1,0],x[yhc==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(x[yhc==2,0],x[yhc==2,1],s=100,c='green',label='cluster 3')
plt.scatter(x[yhc==3,0],x[yhc==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(x[yhc==4,0],x[yhc==4,1],s=100,c='magenta',label='cluster 5')

#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.title('clusters of clients')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()