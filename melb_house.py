import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans  

# download dataframe from kaggle
# link https://www.kaggle.com/dansbecker/melbourne-housing-snapshot

# load dataframe
data = pd.read_csv('melb_data.csv')
# cheking null values in dataframe
data.info()

# Column BuildingArea, YearBuilt, CouncilArea have rows less than 13580
# cheking total null values 
data[['YearBuilt', 'BuildingArea', 'CouncilArea']].isnull().sum()

# Handling missing values for column CouncilArea
modus_CA = data['CouncilArea'].mode()
# Fill column CouncilArea with Moreland
data['CouncilArea'] = data['CouncilArea'].fillna('Moreland')
# For column BuildingArea anc YearBuilt fill with mean for each column
data['YearBuilt'] = data['YearBuilt'].fillna(data['YearBuilt'].mean())
data['BuildingArea'] = data['BuildingArea'].fillna(data['BuildingArea'].mean())


# Cheking again total null values
data[['YearBuilt', 'BuildingArea', 'CouncilArea']].isnull().sum()
# Round numbers in column YearBuilt
data['YearBuilt'] = data['YearBuilt'].apply(lambda x: round(x))


# Exploratory Data Analysis(EDA)
'''
# Count for Type Property
plt.figure(figsize=(10,6))
sns.countplot(x='Type', data=data)
plt.title('Total Property in each Type Property')
plt.xlabel('Type Property')
plt.ylabel('Total Property')

# Making new column Year and Sold_Month from column Date. This column will be used for making lineplot
data['Year'] = pd.to_datetime(data['Date']).dt.year
data['Sold_Month'] = data['Date'].apply(lambda x: datetime.datetime.strptime(x,"%d/%m/%Y").strftime('%Y-%m'))

# Filtering property with Method 'S'
sold = data[data['Method'] == 'S']['Method']
# trend analysis for property sale
plt.figure(figsize=(10,6))
sns.lineplot(x='Sold_Month', y='Price', hue=sold, data=data)
plt.title('Price of Property Sold in Month')
plt.xlabel('Property Sold in Month')
plt.xticks(rotation=45)
plt.ylabel('Price')
plt.legend(title='Method Property', loc='upper left', bbox_to_anchor=(0., 0.5, 0.8, 0.5))

# trend analysis property sale in each type
type_sold = data[data['Method'] == 'S' ]['Type']
plt.figure(figsize=(10,6))
sns.lineplot(x='Sold_Month', y='Price', hue=type_sold, data=data)
plt.title('Price of Property Sold in Month for Each Type ')
plt.xlabel('Property Sold in Month')
plt.xticks(rotation=45)
plt.ylabel('Price')
plt.legend(title='Type Property', loc='upper right', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
plt.show()

# top 10 Property with the highest price
data.sort_values(by='Price', ascending=False).head(10)
# top 10 Property with the cheapest price
data.sort_values(by='Price').head(10)
# property that have distance closest to MCB(Melbourne's Central Business)
data.sort_values(by='Distance').head(10)


# Corelation between avaibility Rooms, Bedroom2 with Price
sns.pairplot(data[['Rooms', 'Bedroom2', 'Price']])
# Corelation between MCB(Melbourne's Central Business) distance from  and price
sns.pairplot(data[['Price', 'Distance']])

# property availability in each regionname
plt.figure(figsize=(10,6))
sns.scatterplot(x='Regionname', y='Propertycount', data=data)
plt.title('Number of Property in each Region')
plt.xticks(rotation=45)

# Corelation between Price and Number of Property
sns.pairplot(data[['Price', 'Propertycount']])'''


#From EDA we get some insight that Price are affected by Avaibility Rooms and Badroom2, Distance from MCB and avaibility of number porperty
#We will do clustering to find price segmentation thay affected some features using K-Means algorithm

# Features/ input for K-Means clustering
X = data[['Rooms', 'Bedroom2', 'Distance', 'Propertycount']]

# make elbow graph to find the optimal number of centroids
inertia = []
for k in range(1,10):
    cluster_model = KMeans(n_clusters=k)
    # fit cluster model to X
    cluster_model.fit(X)
    # Get the inersia value
    inertia_value = cluster_model.inertia_
    #Append the inertia_value to inertia list
    inertia.append(inertia_value)

##Inertia plot
plt.figure(figsize=(12,6))
plt.plot(range(1, 10), inertia)
plt.title('The Elbow Method - Inertia plot', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('inertia')
plt.show()

# Do prediction with number of cluster from elbow method
kmeans_model = KMeans(n_clusters=3, random_state=24)
targets = kmeans_model.fit_predict(X)

# Plot clustering 
import numpy as np
# Convert X to array
X_array = np.array(X)
#Separate X to xs and ys --> use for chart axis
xs = X_array[:, 0]
ys = X_array [:, 1]
# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys, c=targets, alpha=0.5)
# Assign the cluster centers: centroids
centroids = kmeans_model.cluster_centers_
# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D', s=50)
plt.title('K Means Clustering', fontsize = 20)
plt.show()

# Fill result clustering to new column call "Clustering"
data['Clustering'] = targets

# Analysis Clustering using describe() to find min and max number of feature in each Cluster

# Clustering 0
print('Cluster 0')
print(data[data['Clustering'] == 0][['Price','Rooms','Bedroom2','Distance', 'Propertycount', 'Clustering']].describe())

# Clustering 1 
print('Cluster 1')
print(data[data['Clustering'] == 1][['Price','Rooms','Bedroom2','Distance', 'Propertycount', 'Clustering']].describe())

# Clustering 2
print('Cluster 2')
print(data[data['Clustering'] == 2][['Price','Rooms','Bedroom2','Distance', 'Propertycount', 'Clustering']].describe())
