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
Based on the clustering results, it can be observed that the food ingredients exhibit various health levels. The analysis revealed the following patterns:
- Low Calories but High Fat: Ingredients in this category indicating that while they may not contribute significantly to overall calorie intake, they may contain unhealthy fat levels.
- Low Fat but High Calories: Conversely, some ingredients are low in fat but have high calorie counts, suggesting that they could still contribute to excessive calorie consumption even if they appear healthier due to lower fat content.
- Low Calories and Low Fat: This cluster contains ingredients that are generally regarded as healthy options, as they provide lower calorie and fat content.
- High Calories and High Fat: Ingredients in this category are typically considered unhealthy and should be consumed sparingly.

But keep in mind that high-fat or high-calorie foods are not always unhealthy. It depends on the type of fat—healthy fats like unsaturated fats (from avocados, nuts, fish) are good for heart health, while excessive saturated and trans fats are linked to risks. Calorie needs also vary, for example, athletes or people with high energy demands require more calories. A balanced diet with proper nutrients like fiber, vitamins, and minerals is essential. So, fat or calorie content alone doesn't determine if food is unhealthy.
