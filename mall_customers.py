import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore
import streamlit as st

st.title('Hello')
data = pd.read_csv("Mall_Customers.csv")

#print(data.head()) #Preview of the first few rows of a dataset

#Preprocess the Data
# Get a count all NaN values in each column
#print(data.isnull().sum())

#if there are any rows with NaN values, it will be removed
#data_cleaned = data.dropna()

# Convert categorical data to numeric
# In this case we want to represent gender as 1 or 0 
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

#print(data.head())

# Select relevant features for clustering and regression 
features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features 
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

features_scaled_df = pd.DataFrame(features_scaled, columns=['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

#Corr Matrix

# Calculate the correlation matrix
#corr_matrix = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
corr_matrix = features_scaled_df.corr()

# Display the correlation matrix
print(corr_matrix)

# Create a heatmap for better visualization
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Features')
st.pyplot(plt)

#Linear Regression 

# # Define the target variable (here, 'Spending Score (1-100)' is the target)
# X = features_scaled_df[['Annual Income (k$)', 'Gender', 'Age']]
# y = features_scaled_df[['Spending Score (1-100)']]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict the target variable (Spending Score) on the test set
# y_pred = model.predict(X_test)

# Evaluate the model's performance
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")

#Linear regression is not working, so lets go with polynomial regression

#First we will graph out all of the features against the target.

x_feature = features['Annual Income (k$)']
y_target = features['Spending Score (1-100)']

plt.scatter(x_feature, y_target, color='blue', label='Data points')

# Add title and labels
plt.title("Scatter Plot")
plt.xlabel("Feature")
plt.ylabel("Spending Score")

# Display legend
plt.legend()

# Show the plot
plt.show()
st.pyplot(plt)



# X2 = features_scaled_df[['Annual Income (k$)']].values.reshape(-1,1)
# y = features_scaled_df[['Spending Score (1-100)']].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create polynomial features (degree=2 in this case)
# poly = PolynomialFeatures(degree=2)
# X_train_poly = poly.fit_transform(X_train)  # Transform training data
# X_test_poly = poly.transform(X_test)        # Transform test data

# # Make predictions on the test set
# y_pred = model.predict(X_test_poly)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")

# K-Means Clustering works with numerical data, a given number of clusters, when the clusters are equal in size

# To find the best K value (number of clusters) the elbow method can be used, the elbow is the point where adding more cluster starts to only make small differences in WCSS (how tightly grouped the points are)


inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method (Inertia vs K)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()
st.pyplot(plt)

# look at the graph to find the k value
# For this data I chose 5
# Now we can fit the data

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

plt.figure(figsize=(10, 6))

# Plot each cluster in different colors
for cluster in range(3):
    plt.scatter(data[data['Cluster'] == cluster]['Age'], 
                data[data['Cluster'] == cluster]['Spending Score (1-100)'], 
                s=100, label=f'Cluster {cluster}')

plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(plt)



