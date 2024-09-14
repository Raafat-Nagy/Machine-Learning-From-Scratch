import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, K, max_iters=100):
        self.K = K
        self.max_iters = max_iters

        self.centroids = None
        self.clusters = None

    # Step 1: Initialize centroids by selecting K random points
    def initialize_centroids(self, X):
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], self.K, replace=False)
        self.centroids = X[indices]

    # Step 2: Calculate Euclidean distances
    def euclidean_distance(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    # Step 3: Assign clusters based on closest centroids
    def assign_clusters(self, X):
        distances = self.euclidean_distance(X)
        return np.argmin(distances, axis=1)

    # Step 4: Update centroids by calculating the mean of points in each cluster
    def update_centroids(self, X, clusters):
        self.centroids = np.array([X[clusters == k].mean(axis=0) for k in range(self.K)])

    # Full K-Means algorithm
    def fit(self, X):
        self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            self.clusters = self.assign_clusters(X)
            old_centroids = self.centroids.copy()
            self.update_centroids(X, self.clusters)

            # Check for convergence
            if np.all(old_centroids == self.centroids):
                break

    # Step 5: Compute WCSS (Within-cluster sum of squares)
    def compute_wcss(self, X):
        wcss = 0
        for k in range(self.centroids.shape[0]):
            cluster_points = X[self.clusters == k]
            wcss += np.sum((cluster_points - self.centroids[k]) ** 2)
        return wcss

    # Visualization function
    def plot_clusters(self, X):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=self.clusters, cmap="viridis", marker="o", edgecolor="k", s=50)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c="red", marker="X", s=100, label="Centroids")
        plt.title("K-Means Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()