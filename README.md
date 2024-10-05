# Clustering Project

**Project Overview**
This project is about analyzing each ingredient’s macronutrients to fit the customers’ dietary needs. The goal is to create a clustering model to divide ingredients into two clusters: ingredients that are healthy and not healthy, based on macronutrient data (calories, fat, and weight). This project uses the K-Means clustering method to analyze the data.

**Dataset**
**The dataset used in this project was obtained from the Big Data Processing course**
This dataset contains information on different food ingredients, such as their weight, calories, fat, sugar level, and food color.

**Steps in the Clustering Process**
**1. Load Data**
We load the dataset using PySpark, ensuring that the data is correctly read and structured.

df_train = spark.read.csv("Dataset.csv", header=True, inferSchema=True)
df_train.show()


**2. Select Features**
We select relevant features from the dataset: weight, calories, and fat, which are critical for defining ingredient healthiness. Features selection is crucial as it will determine the final outcome.

df_train = df_train.select("weight", "calories", "fat")
df_train.show()


**3. Data Cleaning**
Missing data is handled by dropping rows with null values, ensuring clean data for further processing.

df_train = df_train.dropna()


**4. Data Transformation**
The data requires cleaning to remove units and ensure numerical consistency. We remove units from the weight, calories, and fat columns and cast them as integers.

df_train = df_train.withColumn("weight", regexp_replace(df_train["weight"], " gr", "").cast("int"))
df_train = df_train.withColumn("calories", regexp_replace(df_train["calories"], " cal", "").cast("int"))
df_train = df_train.withColumn("fat", regexp_replace(df_train["fat"], " gr", "").cast("int"))
df_train.show()


**5. Data Normalization**
Before applying K-Means, it’s important to normalize the data so that all features contribute equally to the clustering model. We use StandardScaler for normalization.

df_train = VectorAssembler(inputCols=df_train.columns, outputCol="Vector").transform(df_train)
scaler = StandardScaler(inputCol="Vector", outputCol="features")
df_train = scaler.fit(df_train).transform(df_train)
df_train.show()


**6. Generate Model**
We apply the K-Means algorithm with k=2 to generate the clustering model, which will divide the ingredients into two clusters.

kmeans = KMeans().setK(2)
model = kmeans.fit(df_train).transform(df_train)
df_train.show()


**7. Visualization**
To visualize the clustering, we use Matplotlib to plot the data. The colors represent the clusters formed by the K-Means model.

panda = model.toPandas()
plt.scatter(panda["calories"], panda["fat"], c=panda["prediction"])
plt.xlabel("Calories")
plt.ylabel("Fat")
plt.title("K-Means Clustering (Calories vs Fat)")
plt.show()


# Analysis and Conclusion
K-Means clustering was applied to group ingredients based on calories, fat, and weight, with k=2 representing two clusters that could be interpreted as “healthy” and “not healthy,” though these labels are not explicit.

The choice of features affects clustering outcomes:
Using calories, fat, and weight created more defined clusters.
Including sugar levels led to more overlap between clusters, reducing clarity.

The clusters, visualized in yellow and purple, group ingredients by caloric and fat content. However, without labeled data, we can't definitively interpret the healthiness of each cluster, a common limitation of unsupervised methods like K-Means.

For conclusion, Clustering highlights patterns in nutrient data but doesn't provide a direct answer to whether ingredients are "healthy." A classification model with labeled data would be better suited for answering FitFut’s question on healthiness. K-Means offers valuable insights but is just a starting point.
