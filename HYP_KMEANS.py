from cv2 import kmeans
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score
from math import sqrt
import sys
import Hypmath
import torch
expmap=Hypmath._expmap
project=Hypmath._project
class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 10
        self.plot_figure = True
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]

    def kmeans_plus_plus_centroids(self,X):
        centroids = []  
        centroids.append(list(X[np.random.choice(range(self.num_examples))]))
        for idx in range(self.K-1):
            dist=[]
            for i in range(self.num_examples):
                point=X[i]
                d=sys.maxsize
                for j in range(len(centroids)):
                    tmpdst=distance.euclidean(point,centroids[j])
                    # ds=Hypmath.dist(torch.tensor(point),torch.tensor(centroids[j]))
                    # ds=ds.detach().cpu().numpy()
                    print(tmpdst)
                    d=min(d,tmpdst)

                dist.append(d)
            print(len(dist))    

            dist1=np.array(dist)
            print(dist1)
            nxtcentroid=X[np.argmax(dist1)]
            centroids.append(list(nxtcentroid))
            dist=[]
        # print(centroids)    
        return centroids    
    # def initialize_random_centroids(self, X):
    #     centroids = np.zeros((self.K, self.num_features))
    #     Tempo=X
    #     Tempoindx=[]
    #     for k in range(self.K):
    #         while(1):
    #             hh=np.random.choice(range(self.num_examples))
    #             if hh not in Tempoindx:
    #                 break
    #         centroid = Tempo[hh]
    #         tp= (np.where(Tempo==centroid))
    #         d=np.unique(tp[0])
    #         print(np.unique(tp[0]))
    #         if int(d) in range(1000,2000):
    #             Tempoindx.extend(np.arange(1000,2000))
    #         if int(d) in range(0,1000):
    #             Tempoindx.extend(np.arange(0,1000))
    #         if int(d) in range(2000,3000):
    #             Tempoindx.extend(np.arange(2000,3000))
    #         if int(d) in range(3000,4000):
    #             Tempoindx.extend(np.arange(3000,4000))
    #         if int(d) in range(4000,5000):
    #             Tempoindx.extend(np.arange(4000,5000))
    #         if int(d) in range(5000,6000):
    #             Tempoindx.extend(np.arange(5000,6000))
    #         if int(d) in range(6000,7000):
    #             Tempoindx.extend(np.arange(6000,7000))
    #         if int(d) in range(7000,8000):
    #             Tempoindx.extend(np.arange(7000,8000))
    #         if int(d) in range(8000,9000):
    #             Tempoindx.extend(np.arange(8000,9000))
    #         if int(d) in range(9000,9999):
    #             Tempoindx.extend(np.arange(9000,9999))
    #         centroids[k] = centroid

    #     return centroids

    def create_clusters(self, X, centroids):
       
        clusters = [[] for _ in range(self.K)]

        
        for point_idx, point in enumerate(X):
            lst=[]
            for k in range(self.K):
                lst.append(distance.minkowski(point,centroids[k],2))
                # dstnce=Hypmath.dist(torch.tensor(point),torch.tensor(centroids[k]))
                # dstnce=dstnce.detach().cpu().numpy()
                # lst.append(dstnce)
            closest_centroid=np.argmin(lst)    
                # closest_centroid = np.argmin(
                #     # np.sqrt(np.sum((point - centroids) ** 2, axis=1))
                    
                # )
            clusters[closest_centroid].append(point_idx)
        # print(clusters)
        
        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid

        return centroids

    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    def plot_fig(self, X, y):
        nc=2
        if nc==2:
            pca=PCA(n_components=2)
            tmp = pd.DataFrame(pca.fit_transform(X))
            # print(tmp.head())
            # print(tmp[y==0][0],tmp[y==0][1])
            for i in range(self.K):
                plt.scatter(tmp[y==i][0], tmp[y==i][1],s=20, cmap=plt.cm.Spectral)
            plt.show()
        if nc==3:
            fig = plt.figure(1, figsize=(4, 3))
            plt.clf()
            ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

            plt.cla()

            pca = PCA(n_components=3, random_state=42)
            # pca.fit(X)
            # X = pca.transform(X)
            X = pca.fit_transform(X)
            # y = np.choose(y, [0,1, 2, 3, 4, 5,6,7,8,9]).astype(float)
            p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")
            fig.colorbar(p)
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])

            plt.show()
    def Einstiens_Midpoint(self, clusters, X):
        c=0.9
        centroids=[]
        for cluster in clusters:
            num=[]
            den=[]
            # print(cluster)
            for i in cluster:
                tmp=abs(1-((np.linalg.norm(X[i],2))/c))
                dn=sqrt(tmp)
                lz=1/dn
                den.append(lz)
                nan=[k*lz for k in X[i]]
                num.append(nan)
            sumn=[sum(x) for x in zip(*num)]   
            sumd=sum(den)
            tcen=[j/sumd for j in sumn] 
            centroids.append(tcen)
        return centroids

    def fit(self, X):
        centroids = self.kmeans_plus_plus_centroids(X)

        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)

            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)
            print(len(centroids),len(previous_centroids))
            diff = centroids - previous_centroids
            # print(diff)
            if not diff.any():
                print("Termination criterion satisfied")
                break
            print(f'iteration{it}')

        # Get label predictions
        y_pred = self.predict_cluster(clusters, X)

        if self.plot_figure:
            self.plot_fig(X, y_pred)

        return y_pred

def acc(y_true, y_pred):

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    # for i,j in ind:
    ind=list(ind)
    # print(ind[0])
    # print(i for )
    return (sum([w[i, j] for i, j in zip(ind[0],ind[1])]) * 1.0 / y_pred.size)

if __name__ == "__main__":
    np.random.seed(10)
    num_clusters = 10
    X=np.load('/home/shivanand/Desktop/Minor/Datasets/Latents/idh_latent.npy')
    l=np.load('/home/shivanand/Desktop/Minor/Datasets/Latents/idh_lab1.npy')
    t=project(expmap(torch.tensor(X),torch.tensor(X),0.9),0.9)
    t=t.detach().cpu().numpy()
    Kmeans = KMeansClustering(t, num_clusters)
    # y_pred=KMeans(n_clusters=10).fit_predict(X)
    # print(acc(y_true=l,y_pred=kmeans))
    y_pred = Kmeans.fit(t)
    # print(y_pred)
    print(acc(y_true=l,y_pred=np.array(y_pred,dtype=np.int32)))
    print(normalized_mutual_info_score(l,y_pred))
    print(adjusted_rand_score(l,y_pred))
    print(davies_bouldin_score(X,y_pred))