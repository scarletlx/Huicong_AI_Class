# -*- coding: utf-8 -*-

# KMeans
# step1：随机设定初始聚类中心
# step2：将距离某个聚类中心距离近的样本点归类到该聚类中心，将样本全部归类完毕后得到多个簇
# step3：计算每个簇的均值作为新的聚类中心
# step4：重复第二步和第三步直至聚类中心不再发生变化

# KMeans++
# step1：首先从所有样本中随机选定第一个聚类中心
# step2：记录所有样本到与其最近的聚类中心的距离
# step3：所有非聚类中心样本点被选取作为下一个聚类中心的概率与step2中的距离大小成正比，也就是说距离越远的样本点越有可能成为下一个聚类中心
# step4：重复step2和step3直至选出多个聚类中心

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

class KMcluster():
    def __init__(self, X, y, n_clusters=3, initialize="random", max_iters=10):
        self.X = X
        self.y = y
        self.n_clusters = n_clusters
        self.initialize = initialize
        self.max_iters = max_iters

    # 随机初始化中心点
    def init_random(self):
        # min, max = np.min(self.X)-1, np.max(self.X)+1
        # n_features = self.X.shape[1]
        # centroids = min + (max-min) * np.random.random((self.n_clusters, n_features))
        n_samples, n_features = self.X.shape
        centroids = self.X[np.random.choice(n_samples, 4)]
        return centroids

    # KMeans++ 初始化中心点
    def init_kmeans_plusplus(self):
        n_samples, n_features = self.X.shape

        # step 1: 随机选取第一个中心点
        centroids = self.X[np.random.choice(n_samples, 1)]

        # 计算其余的中心点
        for k in range(0, self.n_clusters-1):
            distances = np.zeros((n_samples, k+1))

            # step 2: 计算每个样本到每一个聚类中心的欧式距离
            for i in range(len(centroids)):
                distances[:, i] = np.sqrt(np.sum(np.square(self.X - centroids[i]), axis=1))

            # step 3: 计算每个样本与最近聚类中心(指已选择的聚类中心)的距离D(x)
            dist = np.min(distances, axis=1)

            # step 4: 再取一个随机值，用权重的方式来取计算下一个“种子点”。具体实现是，先取一个能落在Sum(D(x))中的随机值Random，
            # 然后用Random -= D(x)，直到其<=0，此时的点就是下一个“种子点”。
            total = np.sum(dist) * np.random.rand()
            for j in range(n_samples):
                total -= dist[j]
                if total > 0:
                    continue
                centroids = np.r_[centroids, self.X[j].reshape(-1, 2)]
                break

        # print(centroids)
        return centroids

    def assignment(self, centroids):
        n_samples = self.X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        for i in range(self.n_clusters):
            distances[:,i] = np.sum(np.square(self.X - centroids[i]), axis=1)
        return np.argmin(distances, axis=1)

    def update_center(self, flag, centroids):
        new_centroids = np.zeros_like(centroids)
        for i in range(self.n_clusters):
            new_centroids[i] = np.mean(self.X[flag==i], axis=0)
        return new_centroids

    def train(self):
        # step 1: generate center
        if self.initialize == "kmeans++":
            centroids = self.init_kmeans_plusplus()
        else:
            centroids = self.init_random()

        colmap = [i for i in range(self.n_clusters)]
        for i in range(self.max_iters):
            # step 2: assign centroid for each source data
            flag = self.assignment(centroids)

            plt.scatter(self.X[:,0], self.X[:,1], c=flag, marker=".", alpha=0.5)
            plt.scatter(centroids[:, 0], centroids[:, 1], c=colmap, marker="o", linewidths=6)
            plt.show()

            # step 3: re-caculate center
            new_centroids = self.update_center(flag, centroids)

            # 终止条件，如果重新计算的中心点与上一次的重复，则退出训练
            if (new_centroids == centroids).all():
                break
            else:
                centroids = new_centroids

            print("iters: ", i, ", center point: ", centroids)


if __name__=="__main__":

    # 生成数据集: X[2000, 2], y[2000], 4 clusters
    X, y = sklearn.datasets.make_blobs(n_samples=2000,
                                       n_features=2,
                                       centers=4,
                                       random_state=40,
                                       cluster_std=(2.5, 2, 2.5, 2))

    km = KMcluster(X, y, n_clusters=4, initialize="random", max_iters=50)
    # km = KMcluster(X, y, n_clusters=4, initialize="kmeans++", max_iters=50)
    km.train()
