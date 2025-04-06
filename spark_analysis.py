from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, when
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("SentimentAnalysisVisualization") \
    .getOrCreate()

# Load the dataset from HDFS or local
file_path = "file:///mnt/e/bda_lab/bda_project/all_data.csv"
df = spark.read.option("header", True).csv(file_path)

# Convert data types
df = df.withColumn("Likes", col("Likes").cast("int")) \
       .withColumn("Sentiment_score", col("Sentiment_score").cast("float"))

# Define sentiment categories
df = df.withColumn(
    "Sentiment_Label",
    when(col("Sentiment_score") > 0.05, "Positive")
    .when(col("Sentiment_score") < -0.05, "Negative")
    .otherwise("Neutral")
)

# Aggregated statistics
df.groupBy("Sentiment_Label").agg(
    count("*").alias("count"),
    avg("Sentiment_score").alias("avg_sentiment_score"),
    avg("Likes").alias("avg_likes")
).show()

# Convert to Pandas for visualization
pdf = df.select("Sentiment_Label", "Sentiment_score", "Likes").toPandas()

# Set style for plots
sns.set(style="whitegrid")

# 1. Pie chart - Sentiment distribution
plt.figure(figsize=(6, 6))
sentiment_counts = pdf["Sentiment_Label"].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["green", "gray", "red"])
plt.title("Sentiment Distribution")
plt.show()

# 2. Bar chart - Average likes per sentiment
plt.figure(figsize=(8, 6))
avg_likes = pdf.groupby("Sentiment_Label")["Likes"].mean()
avg_likes.plot(kind='bar', color=["green", "gray", "red"])
plt.ylabel("Average Likes")
plt.title("Average Likes by Sentiment")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 3. Histogram - Sentiment score distribution
plt.figure(figsize=(8, 6))
plt.hist(pdf["Sentiment_score"], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 4. Scatter Plot - Likes vs Sentiment Score
plt.figure(figsize=(10, 6))
plt.scatter(pdf["Sentiment_score"], pdf["Likes"], alpha=0.5, color='purple')
plt.title("Likes vs Sentiment Score")
plt.xlabel("Sentiment Score")
plt.ylabel("Likes")
plt.grid(True)
plt.show()

# Stop Spark Session
spark.stop()
