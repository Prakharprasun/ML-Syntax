from sklearn.cluster import KMeans
wcss_scores = []

for i in range(1,11):
    kmean_model = KMeans(n_clusters=i, n_init='auto') # create a KMeans model with i clusters
    kmean_model.fit(clustering_data) # fit the model to the data
    wcss_scores.append(kmean_model.inertia_) # get the WCSS score and append it to the list. the inertia_ attribute gives the WCSS score

np.array(wcss_scores) # convert the list to a numpy array to use it in the plot

# plot the WCSS scores
plt.plot(range(1,11),wcss_scores)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS Scores')
plt.show()

kmean_model = KMeans(n_clusters=3, n_init='auto') # create a KMeans model with 3 clusters
kmean_model.fit(clustering_data) # fit the model to the data

clusters = clustering_data.copy() # create a copy of the data
clusters["Predicted Class"] = kmean_model.fit_predict(clustering_data) # add a new column to the data with the predicted class
clusters.head()

kmean_model.cluster_centers_ # get the cluster centers

kmean_model.cluster_centers_.shape

#Plot the different class with different colours. The c= controls the colours and the label= controls the name shown in the legend
plt.scatter(df_Setosa['petal_length'],df_Setosa['petal_width'],c='r',label='Setosa')
plt.scatter(df_Versicolor['petal_length'],df_Versicolor['petal_width'],c='b',label='Versicolor')
plt.scatter(df_Verginica['petal_length'],df_Verginica['petal_width'],c='g',label='Verginica')

#Plot the cluster centers
plt.scatter(kmean_model.cluster_centers_[:,2],kmean_model.cluster_centers_[:,3],marker="*",c='yellow',edgecolor="black",label='Cluster Centers', s=200) # the cluster centers for petals are the last 2 columns of the cluster_centers_ attribute

#Add in the axis and legend
plt.legend()
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

plt.scatter(df_Setosa['sepal_length'],df_Setosa['sepal_width'],c='r',label='Setosa')
plt.scatter(df_Versicolor['sepal_length'],df_Versicolor['sepal_width'],c='b',label='Versicolor')
plt.scatter(df_Verginica['sepal_length'],df_Verginica['sepal_width'],c='g',label='Verginica')

plt.scatter(kmean_model.cluster_centers_[:,0],kmean_model.cluster_centers_[:,1],marker="*",c='yellow',edgecolor="black",label='Cluster Centers', s=200) # the cluster centers for petals are the last 2 columns of the cluster_centers_ attribute

#Add in the axis and legend
plt.legend()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.show()