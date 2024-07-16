# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the dataset (assuming it's saved as 'customer_data.csv' in the current directory)
df = pd.read_csv('marketing_campaign.csv')

# Data preprocessing
# Replace missing values, encode categorical variables, etc.
# For simplicity, assuming the dataset is already preprocessed

# Separate features and target variable
X = df.drop(['target_variable'], axis=1)  # Replace 'target_variable' with your actual target column
y = df['target_variable']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)  # Assuming we want to reduce to 2 principal components
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame for the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Optional: Explore explained variance ratio to determine the number of components
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance)

# Perform clustering (example using K-means)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(principal_df[['PC1', 'PC2']])

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(principal_df['PC1'], principal_df['PC2'], c=clusters, cmap='viridis', alpha=0.5)
plt.title('PCA Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
