# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Hopper Puzzle
# Hopper provided a dataset for their Senior Data Analyst post, so I thought to explore what the data can tell us.
# (Source: http://bit.ly/2q6U8dq)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# %% [markdown]
# ### Starting off; Imports, Data Exploration
# First we'll import the dataset (saved locally to the github for illustration), we could either open it up briefly in excel (if the file is small), or get python to show us a sample. Starting off on any data project is to review what we've got, for completeness and to see what we're working with.
# 
# Below I'll use panda's `.head()`, `.shape` and `.info()` to see what we're working with.

# %%
source = r"D:\Downloads\puzzle.xlsx"
df_puzzle = pd.read_excel(source,header=None)

df_puzzle.head()


# %%
df_puzzle.shape


# %%
df_puzzle.info()

# %% [markdown]
# ### Initial Observations
# In summary, what we've got here is a 1024 point, two dimension dataset, no blanks, and no labels. Without labels, this tells me that we have no classification, so if we do any machine learning, we should use an unsupevised approach.  
# 
# Next, lets plot this to see what we're working with (what the data looks like).

# %%

x = df_puzzle[0]
y = df_puzzle[1]

plt.plot(x, y,'.')
plt.show

# %% [markdown]
# Cool. Looks like we have some clusters going on here. Since we have no labels, let use k-means for this example; k-means is an unsupervised approach of machine learning, we don't need test/training datasets or labels. I also choose k-means because initial plotting we seem to be seeing clusters. We could also apply regression if these are in fact non-discrete datapoints. 
# 
# Since we'll explore this by plotting, before we go into the deep end, lets see what we get for optimal number of clusters. I'll demonstrate using the average silouette method.
# %% [markdown]
# ### Average Silouette

# #%%
# range_n_clusters = [2,3,4,5,6]
# for n_clusters in range_n_clusters:
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)

#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(x)

#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(x, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)

#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(x, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]

#         ith_cluster_silhouette_values.sort()

#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i

#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)

#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples

#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")

#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

#     # 2nd Plot showing the actual clusters formed
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')

#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')

#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                     s=50, edgecolor='k')

#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")

#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')

# plt.show()

# %%
def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(df_puzzle.shape[0])[:n_clusters]
    centers = df_puzzle[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(df_puzzle, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([df_puzzle[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

centers, labels = find_clusters(df_puzzle, 4)
plt.scatter(x,y, c=labels,
            s=50, cmap='viridis');
#%%
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(df_puzzle)
# y_kmeans = kmeans.predict(df_puzzle)

# plt.scatter(df_puzzle[0],df_puzzle[1],c=y_kmeans,s=50,cmap='viridis')

# centers = kmeans.cluster_centers_
# plt.scatter(centers[0],centers[1],c='black',s=200,alpha=0.5)

# %% [markdown]
# ## Conclusion
# And there we have it. `n` number of clusters plotted from a random dataset. It'll be interesting to see what these two columns actually were or meant! Thanks @hopper for providing the data.

# %%



