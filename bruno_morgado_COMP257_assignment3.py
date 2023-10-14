#!/usr/bin/env python
# coding: utf-8

# # Hierarchical Clustering
# ### Bruno Morgado (301154898)

# In[1]:


# Necessary imports
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import AgglomerativeClustering
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[ ]:





# In[3]:


# Fetch Olivetti dataset from Sklearn
dataset = fetch_olivetti_faces(shuffle=True, random_state=98)


# In[4]:


# Storing features, target variable, and 2d features matrix as images
X = dataset.data
y = dataset.target
images = dataset.images
labels = dataset.target


# In[5]:


images.shape


# In[6]:


# Bundle X and y into a dataframe
pixel_columns = [f"pixel_{i}" for i in range(1, X.shape[1] + 1)]

df = pd.DataFrame(X, columns=pixel_columns)

df['target'] = y


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


# Split dataset into train, validation, and test sets with stratification
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=98, stratify=y)

X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=98, stratify=y_temp)

print(f"Training set size: {len(y_train)}")
print(f"Validation set size: {len(y_valid)}")
print(f"Test set size: {len(y_test)}")


# In[10]:


# train the Logistic Regression Classifier
log_clf = LogisticRegression('l2', random_state=98)


# In[11]:


# Get 5-fold cross validation scores
k = 5
scores = cross_val_score(log_clf, X_train, y_train, cv=k, scoring='accuracy')

print(f"Cross-validation scores (k={k}):", scores)
print("Average cross-validation score:", scores.mean())


# In[12]:


# Train the Logistic Regression classifier
log_clf.fit(X_train, y_train)


# In[13]:


# Make predictions and print validation scores on the validation set
y_pred_valid = log_clf.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred_valid)
print(f"Validation accuracy: ", accuracy)


# In[14]:


# Make predictions on the test set
y_pred = log_clf.predict(X_test)


# In[15]:


# Print the classification report
print('\t\tClassification Report - Logistic Regression\n\n', classification_report(y_test, y_pred))


# In[16]:


desired_images = images[labels == 7]


# In[17]:


def plot_images(images):

    n_images = len(images)
    rows = int(np.sqrt(n_images))
    cols = (n_images // rows) + int(n_images % rows != 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    
    # Plot all images
    for ax, img in zip(axes.ravel(), images):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    
    # Turn off any remaining axes
    for ax in axes.ravel()[n_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# In[18]:


plot_images(desired_images)


# In[19]:


n_clusters_range = range(2, 201)
def compute_scores(X, metric, linkage_method):
    scores = []
    for n_clusters in n_clusters_range:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity=metric, linkage=linkage_method)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)
        scores.append(silhouette_avg)
    return scores


# In[20]:


# Calculate silhouette scores for each metric
euclidean_scores = compute_scores(X, 'euclidean', 'ward')
minkowski_scores = compute_scores(X, 'minkowski', 'average')  # Using average linkage with Minkowski
cosine_scores = compute_scores(X, 'cosine', 'average')  # Using average linkage with Cosine


# In[21]:


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, euclidean_scores, label='Euclidean Distance')
plt.plot(n_clusters_range, minkowski_scores, label='Minkowski Distance')
plt.plot(n_clusters_range, cosine_scores, label='Cosine Similarity')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.legend()
plt.title('Silhouette Scores for Different Similarity Measures')
plt.show()


# In[22]:


best_n_clusters_euclidean = n_clusters_range[euclidean_scores.index(max(euclidean_scores))]
best_n_clusters_euclidean


# In[23]:


euclidean_scores[129]


# In[24]:


best_n_clusters_minkowski = n_clusters_range[minkowski_scores.index(max(minkowski_scores[15:]))]
best_n_clusters_minkowski


# In[25]:


euclidean_scores[159]


# In[26]:


best_n_clusters_cosine = n_clusters_range[cosine_scores.index(max(cosine_scores[15:]))]
best_n_clusters_cosine


# In[28]:


# Initialize a dictionary to store the reduced data
reduced_data = {}

# Metrics to perform dimensionality reduction
metrics = ['euclidean', 'minkowski', 'cosine']

for metric in metrics:
    # Here, 'ward' linkage only works for the 'euclidean' metric
    linkage = 'ward' if metric == 'euclidean' else 'average'
    
    if metric == 'euclidean':
        n_clusters = best_n_clusters_euclidean
    elif metric == 'minkowski':
        n_clusters = best_n_clusters_minkowski
    else:
        n_clusters = best_n_clusters_cosine
    
    # Perform AHC clustering
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=metric, linkage=linkage)
    labels = agg_cluster.fit_predict(X)
    
    # Compute centroids for each cluster
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
    
    # Replace data points with their cluster centroid values
    reduced_X = centroids[labels]
    
    reduced_data[metric] = reduced_X

# The dictionary 'reduced_data' now has the compressed datasets for each metric
print(reduced_data['euclidean'].shape) 
print(reduced_data['minkowski'].shape)
print(reduced_data['cosine'].shape)


# In[29]:


# Split Compressed Euclidean dataset into train, validation, and test sets with stratification
X_train_eu, X_temp_eu, y_train_eu, y_temp_eu = train_test_split(reduced_data['euclidean'], y, test_size=0.2, random_state=98, stratify=y)

X_valid_eu, X_test_eu, y_valid_eu, y_test_eu = train_test_split(X_temp_eu, y_temp_eu, test_size=0.5, random_state=98, stratify=y_temp_eu)

print(f"Training set size: {len(y_train_eu)}")
print(f"Validation set size: {len(y_valid_eu)}")
print(f"Test set size: {len(y_test_eu)}")


# In[30]:


# train the Logistic Regression Classifier
log_clf_eu = LogisticRegression('l2', random_state=98)


# In[31]:


# Get 5-fold cross validation scores for the Euclidean compressed set
k = 5
scores = cross_val_score(log_clf_eu, X_train_eu, y_train_eu, cv=k, scoring='accuracy')

print(f"Cross-validation scores (k={k}):", scores)
print("Average cross-validation score:", scores.mean())


# In[32]:


# Split Compressed Minkowski dataset into train, validation, and test sets with stratification
X_train_mi, X_temp_mi, y_train_mi, y_temp_mi = train_test_split(reduced_data['minkowski'], y, test_size=0.2, random_state=98, stratify=y)

X_valid_mi, X_test_mi, y_valid_mi, y_test_mi = train_test_split(X_temp_mi, y_temp_mi, test_size=0.5, random_state=98, stratify=y_temp_mi)

print(f"Training set size: {len(y_train_mi)}")
print(f"Validation set size: {len(y_valid_mi)}")
print(f"Test set size: {len(y_test_mi)}")


# In[33]:


# train the Logistic Regression Classifier
log_clf_mi = LogisticRegression('l2', random_state=98)


# In[34]:


# Get 5-fold cross validation scores for the Minkowski compressed set
k = 5
scores = cross_val_score(log_clf_mi, X_train_mi, y_train_mi, cv=k, scoring='accuracy')

print(f"Cross-validation scores (k={k}):", scores)
print("Average cross-validation score:", scores.mean())


# In[35]:


# Split Compressed Minkowski dataset into train, validation, and test sets with stratification
X_train_co, X_temp_co, y_train_co, y_temp_co = train_test_split(reduced_data['cosine'], y, test_size=0.2, random_state=98, stratify=y)

X_valid_co, X_test_co, y_valid_co, y_test_co = train_test_split(X_temp_co, y_temp_co, test_size=0.5, random_state=98, stratify=y_temp_co)

print(f"Training set size: {len(y_train_co)}")
print(f"Validation set size: {len(y_valid_co)}")
print(f"Test set size: {len(y_test_co)}")


# In[36]:


# train the Logistic Regression Classifier
log_clf_co = LogisticRegression('l2', random_state=98)


# In[37]:


# Get 5-fold cross validation scores for the Cosine compressed set
k = 5
scores = cross_val_score(log_clf_co, X_train_co, y_train_co, cv=k, scoring='accuracy')

print(f"Cross-validation scores (k={k}):", scores)
print("Average cross-validation score:", scores.mean())


# # END

# 
