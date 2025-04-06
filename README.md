# Project Overview
This project focuses on performing real-time sentiment analysis on live social media data gathered from Bluesky. It integrates multiple big data and machine learning technologies, including Apache Spark, Kafka, HDFS, MLLib, and the Bluesky API, to build a pipeline that captures posts based on a user query and analyzes their sentiment in real-time.
# Keywords and Main Features
# Real-time Data Ingestion using Bluesky API & Kafka
The system starts by querying the Bluesky API to fetch recent posts based on user-defined keywords. These posts are streamed in real-time using Apache Kafka, ensuring low-latency data flow for continuous sentiment monitoring.
# Distributed Storage using HDFS
All the incoming posts are stored in the Hadoop Distributed File System (HDFS), enabling scalable and fault-tolerant data storage for both live processing and historical analysis.
# Processing with Apache Spark
Apache Spark is used for distributed data processing. It consumes the streamed data from Kafka, processes it in micro-batches, and prepares it for sentiment analysis using Spark Streaming.
# Sentiment Analysis with MLLib
Spark’s MLLib library is employed to perform sentiment classification. The posts are vectorized, and a machine learning model is used to classify them into positive, negative, or neutral sentiments.

# Scalable and Real-Time Insights
The integration of these tools allows for real-time sentiment insights on trending topics, which can be extended for use cases in brand monitoring, opinion tracking, or public sentiment measurement at scale.
