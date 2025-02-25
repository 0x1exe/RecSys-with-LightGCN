from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
import random
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1,org.postgresql:postgresql:42.7.5 pyspark-shell'

spark = SparkSession.builder \
    .appName("KafkaToPostgresProcessing") \
    .config("spark.jars", "postgresql-42.7.5.jar") \
    .getOrCreate()

schema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("item_id", IntegerType(), True),
    StructField("rating", IntegerType(), True),
    StructField("timestamp", LongType(), True)
])

df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "127.0.0.1:9092") \
    .option("subscribe", "user_interactions") \
    .load()

parsed_df = df \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

seed = 16
random.seed(seed)

def process_batch(batch_df, batch_id):  
    if batch_df.isEmpty():
        print(f"Batch {batch_id} is empty")
        return
    
    train_df, test_df = batch_df.randomSplit([0.8, 0.2], seed=seed)
    
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_idx").setHandleInvalid("skip")
    item_indexer = StringIndexer(inputCol="item_id", outputCol="item_id_idx").setHandleInvalid("skip")
    
    user_indexer_model = user_indexer.fit(train_df)
    item_indexer_model = item_indexer.fit(train_df)
    
    train_df = user_indexer_model.transform(train_df)
    train_df = item_indexer_model.transform(train_df)
    
    train_user_ids = [row.user_id for row in train_df.select("user_id").distinct().collect()]
    train_item_ids = [row.item_id for row in train_df.select("item_id").distinct().collect()]
    
    test_df = test_df.filter(
        col("user_id").isin(train_user_ids) & 
        col("item_id").isin(train_item_ids)
    )
    
    test_df = user_indexer_model.transform(test_df)
    test_df = item_indexer_model.transform(test_df)
    print(f'Train df processed: {train_df.count()}')
    train_df.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://127.0.0.1:5432/movie_lens") \
        .option("dbtable", "train") \
        .option('user','postgres') \
        .option('password','psw689') \
        .option("driver", "org.postgresql.Driver") \
        .mode("append") \
        .save()
    
    test_df.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://127.0.0.1:5432/movie_lens") \
        .option("dbtable", "test") \
        .option("user", "postgres") \
        .option("password", "psw689") \
        .option("driver", "org.postgresql.Driver") \
        .mode("append") \
        .save()

query = parsed_df \
    .writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option("isolationLevel","READ_COMMITTED") \
    .option("checkpointLocation", "./tmp/checkpoint")\
    .start()

query.awaitTermination()

spark.stop()
