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

data = pd.read_csv("Housing.csv")

print(data.head())

numeric_data = data.select_dtypes(include=['number'])

print(numeric_data.columns)

features = numeric_data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

features_scaled_df = pd.DataFrame(features_scaled)
#Corr Matrix

corr_matrix = features_scaled_df.corr()

# Display the correlation matrix
print(corr_matrix)

# Create a heatmap for better visualization
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()