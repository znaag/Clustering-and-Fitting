# -*- coding: utf-8 -*-
# Import needed libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

import seaborn as sns
sns.set()

pio.renderers.default = 'notebook_connected'

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings('ignore')

# Load in the datasets

df = pd.read_csv('./content/Dataset.csv',skiprows=4)
dfCountry = pd.read_csv('./content/Countries_Detail.csv')

# Rename columns in the country dataframe
dfCountry = dfCountry.rename(columns={'name':'Country Name', 'alpha-3':'Country Code','country_code':'Country_Code', 'region':'Continent'}, inplace=False)

# Select only needed columns
dfCountry_1 = dfCountry.drop(['country-code', 'alpha-2', 'iso_3166-2', 'sub-region', 'intermediate-region', 'region-code', 'sub-region-code', 'intermediate-region-code'], axis=1)

# Drop missing observations
dfCountry_1 = dfCountry_1.dropna(inplace=False)

# Merge the two dataframes
df_merged = df.merge(dfCountry_1, on=['Country Name', 'Country Code'])

# Pick an indicator from the list of indicators
df_CO2 = df_merged[df_merged['Indicator Name'] == 'CO2 emissions from liquid fuel consumption (% of total)']
df_CO2.head()

# Melt the dataframe
df_CO2_melted = df_CO2.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code','Continent',], var_name='Year', value_name='CO2 emissions from liquid fuel consumption (% of total)')

# Drop unneeded columns
df_CO2_melted_1 = df_CO2_melted.drop(['Indicator Name', 'Indicator Code', ], axis=1)

# Drop missing observations
df_CO2_melted_2 = df_CO2_melted_1.dropna(inplace=False)
df_CO2_melted_2.head()

# Create a function to encode the categorical (or non-numeric) variable into numerical values

def encode_label(df):
   
    le = LabelEncoder()

    df_CO2_melted_2["Country_le"] = le.fit_transform(df_CO2_melted_2["Country Name"])
    df_CO2_melted_2['Continent_le'] = le.fit_transform(df_CO2_melted_2['Continent'])

encode_label(df_CO2_melted_2)


# Drop continent off the dataframe
x = df_CO2_melted_2.drop(['Continent'], axis=1)
X = x.iloc[:, 2:]

wcss = [] 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('WCSS Chart for Clusters', fontweight='bold')
plt.show()

# Create a function for fitting the clustering model

def clustering(df):
    km = KMeans(n_clusters=4)
    km

    y_predicted = km.fit_predict(X[["Country_le","Year"]])
    
    return y_predicted
X['Clustered'] = clustering(X)

# Merge dataframes
data = X.merge(df_CO2_melted_2, on=['Continent_le', 'Country_le', 'Year', 'CO2 emissions from liquid fuel consumption (% of total)'])

data_grouped = data.groupby(['Country Name', 'Continent', 'Year']).mean().reset_index()
data_grouped = data_grouped.drop(['Country_le', 'Continent_le'], axis=1)
data_grouped.head()

# Round clustered value to the nearest whole number
data_grouped['Clustered'] = data_grouped['Clustered'].round(0)

fig = px.scatter(data_grouped, x='Country Name', y= 'Continent', color='Clustered',
                hover_data=['CO2 emissions from liquid fuel consumption (% of total)', 'Country Name', 'Continent'], 
                title='Scatter Plot of Clustered Data')
fig.write_image("/content/scatter.png", format="png", width=1200, height=800, engine="kaleido")

#cluster1
data_grouped_comparative = data_grouped[data_grouped['Year'].between('2013', '2018')]
df_cluster_1 = data_grouped_comparative[data_grouped_comparative['Clustered'] == 0]
df_cluster_1_grouped = df_cluster_1.groupby(['Year', 'Country Name', 'Continent']).mean().reset_index()
fig = px.scatter(df_cluster_1_grouped, x='Year', y='CO2 emissions from liquid fuel consumption (% of total)', color='Continent',
                hover_data=['Country Name'], title='Scatter Plot of Cluster 1')
fig.write_image("/content/cluster1.png", format="png", width=1200, height=800, engine="kaleido")

#cluster2
df_cluster_2 = data_grouped_comparative[data_grouped_comparative['Clustered'] == 1]
df_cluster_2_grouped = df_cluster_2.groupby(['Year', 'Country Name', 'Continent']).mean().reset_index()
fig = px.scatter(df_cluster_2_grouped, x='Year', y='CO2 emissions from liquid fuel consumption (% of total)', color='Continent',
                hover_data=['Country Name'], title='Scatter Plot of Cluster 2')
fig.write_image("/content/cluster2.png", format="png", width=1200, height=800, engine="kaleido")

#cluster3
df_cluster_3 = data_grouped_comparative[data_grouped_comparative['Clustered'] == 2]
df_cluster_3_grouped = df_cluster_3.groupby(['Year', 'Country Name', 'Continent']).mean().reset_index()
fig = px.scatter(df_cluster_3_grouped, x='Year', y='CO2 emissions from liquid fuel consumption (% of total)', color='Continent',
                hover_data=['Country Name'], title='Scatter Plot of Cluster 3')
fig.write_image("/content/cluster3.png", format="png", width=1200, height=800, engine="kaleido")

#cluster4
df_cluster_4 = data_grouped_comparative[data_grouped_comparative['Clustered'] == 3]                 
df_cluster_4_grouped = df_cluster_4.groupby(['Year', 'Country Name', 'Continent']).mean().reset_index()                      
fig = px.scatter(df_cluster_4_grouped, x='Year', y='CO2 emissions from liquid fuel consumption (% of total)', color='Continent',
                hover_data=['Country Name'], title='Scatter Plot of Cluster 4')
fig.write_image("/content/cluster4.png", format="png", width=1200, height=800, engine="kaleido")

#curve fit
def func(X , a , b):
  return a+b*X

data_compared_1 = data_grouped.pivot(index="Country Name", columns="Year", values="CO2 emissions from liquid fuel consumption (% of total)")                                                                  
XData = data_compared_1['2011']
YData = data_compared_1['2016']

fig = px.scatter(data_compared_1.reset_index(), x='2011', y='2016', hover_data=['Country Name'],
                title='Scatter Plot of Fitted Data')
fig.write_image("/content/scatter_Curve_fit.png", format="png", width=1200, height=800, engine="kaleido")

# Drop missing observations

XData = XData.dropna(inplace=False)
YData = YData.dropna(inplace=False)

# Ensure both XData and YData have the same length
min_length = min(len(XData), len(YData))
XData = XData[:min_length]
YData = YData[:min_length]

# Curve Fit

popt , pcov = curve_fit(func , XData , YData )

fitting = np.arange(0.0,0.09,0.001)
plt.plot(fitting, func(fitting,*popt),'r')
plt.xlabel('2011')
plt.ylabel('2016')
plt.title('Fit Curve',fontweight='bold')
plt.savefig("/content/Curvefit.png", format="png", dpi=300, bbox_inches='tight')


