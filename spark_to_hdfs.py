from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, current_timestamp, struct
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType
from nltk.sentiment import SentimentIntensityAnalyzer
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import nltk
import re

# Download VADER Lexicon (only needed once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('stemmer/porter.pickle')
except LookupError:
    nltk.download('punkt')  # stemmer doesn't need downloading, tokenizer does

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Initialize SparkSession with optimizations for HDFS
spark = SparkSession.builder \
    .appName("BlueskyStreamProcessor") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define JSON Schema for Bluesky posts
json_schema = StructType() \
    .add("query", StringType()) \
    .add("text", StringType()) \
    .add("author", StringType()) \
    .add("likes", IntegerType())

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()
stemmer = PorterStemmer()
# Preprocessing function
def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)            # Remove URLs
    text = re.sub(r"@\w+", "", text)                               # Remove @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                          # Remove # from hashtags
    text = re.sub(r"[\r\n]+", " ", text)                           # Remove newlines
    text = re.sub(r"[^\w\s]", "", text)                            # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)                     # Remove non-alphanumeric chars
    text = re.sub(r"\s+", " ", text).strip()                       # Normalize whitespace

    # Tokenization and Stemming
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

# Sentiment Analysis UDF with Preprocessing
@pandas_udf(FloatType(), PandasUDFType.SCALAR)
def analyze_sentiment_udf(text_series: pd.Series) -> pd.Series:
    return text_series.apply(lambda x: sia.polarity_scores(x)["compound"] if x else 0.0)

# Register UDFs
sentiment_udf = analyze_sentiment_udf
preprocess_udf = udf(preprocess_text, StringType())

# Read from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "bluesky_search") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Process Kafka Data
raw_df = kafka_df.selectExpr("CAST(value AS STRING)")

parsed_df = raw_df.withColumn("parsed_json", from_json(col("value"), json_schema)) \
                   .select(col("parsed_json.*"))

# Clean and preprocess text fields
cleaned_df = parsed_df.withColumn("clean_text", preprocess_udf(col("text"))) \
                       .withColumn("clean_query", preprocess_udf(col("query")))

# Add sentiment score and timestamp
processed_df = cleaned_df.withColumn("sentiment_score", sentiment_udf(col("clean_text"))) \
                         .withColumn("ingest_timestamp", current_timestamp())
final_df = processed_df.select(
    "author",
    "likes",
    "clean_text",
    "clean_query",
    "sentiment_score",
    "ingest_timestamp"
)
# Write to HDFS in CSV format
hdfs_query = final_df.writeStream \
    .format("csv") \
    .option("path", "hdfs://localhost:9000/user/spark/bluesky_data") \
    .option("checkpointLocation", "hdfs://localhost:9000/user/spark/checkpoints") \
    .option("header", "true") \
    .option("delimiter", ",") \
    .outputMode("append") \
    .start()

hdfs_query.awaitTermination()