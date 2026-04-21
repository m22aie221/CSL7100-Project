from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
from config import PATHS, USE_HDFS
from pyspark.sql.functions import col, rand, row_number
from pyspark.sql.window import Window

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rand, col

from pyspark.sql.functions import rand

def user_based_split(df, test_ratio=0.2):

    print("🔀 Performing user-based split (corrected)")

    df = df.withColumn("rand", rand(seed=42))

    # Ensure each user has at least 1 train sample
    window = Window.partitionBy("user_id").orderBy(col("rand"))
    df = df.withColumn("row_num", row_number().over(window))

    # First interaction → always train
    train_df = df.filter(
        (col("rand") > test_ratio) | (col("row_num") == 1)
    ).drop("rand", "row_num")

    test_df = df.filter(
        (col("rand") <= test_ratio) & (col("row_num") != 1)
    ).drop("rand", "row_num")

    print(f"Train: {train_df.count()}, Test: {test_df.count()}")

    return train_df, test_df

    

def create_spark_session(app_name="RecommenderGraph-ETL"):
    builder = SparkSession.builder.appName(app_name)

    if USE_HDFS:
        builder = builder.config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
        print("🚀 Running in HDFS mode")
    else:
        builder = (
            builder.config("spark.hadoop.fs.defaultFS", "file:///")
            .config("spark.driver.memory", "5g")
            .config("spark.executor.memory", "5g")
            .config("spark.sql.shuffle.partitions", "200")   # 🔥 important
            .config("spark.default.parallelism", "200")
        )
        print("💻 Running in LOCAL mode")

    return builder.getOrCreate()


def load_raw_data(spark):
    input_path = PATHS["raw"]
    print(f"📥 Loading data from: {input_path}")

    df = spark.read.json(input_path)

    # Optional sampling for development
    df = df.sample(0.2, seed=42)

    print("📊 Schema:")
    df.printSchema()

    return df



from pyspark.sql.functions import col, regexp_replace

def select_relevant_columns(df):
    df = df.select(
        col("reviewerID").alias("user_id"),
        col("asin").alias("item_id"),
        col("overall").alias("rating"),
        col("unixReviewTime").alias("timestamp"),
        col("verified"),
        col("vote"),
        col("reviewText"),
        col("summary")
    )

    # Convert vote from string ("1,234") → int
    df = df.withColumn(
        "vote",
        regexp_replace(col("vote"), ",", "").cast("int")
    )

    return df

def filter_active_users_items(df, min_user_interactions=5, min_item_interactions=5):
    from pyspark.sql.functions import count

    user_counts = df.groupBy("user_id").agg(count("*").alias("user_count"))
    item_counts = df.groupBy("item_id").agg(count("*").alias("item_count"))

    df = df.join(user_counts, "user_id")
    df = df.join(item_counts, "item_id")

    df = df.filter(
        (col("user_count") >= min_user_interactions) &
        (col("item_count") >= min_item_interactions)
    )

    return df.drop("user_count", "item_count")    

def clean_data(df):
    df = df.dropna(subset=["user_id", "item_id", "rating"])

    # Optional: remove unverified reviews
    df = df.filter((col("verified") == True) | col("verified").isNull())

    # Fill missing votes
    df = df.fillna({"vote": 0})

    print("✅ Data cleaned")
    return df



def save_train_test(train_df, test_df):
    base_path = PATHS["parquet"]

    train_path = base_path + "/train"
    test_path = base_path + "/test"

    print(f"💾 Saving TRAIN to: {train_path}")
    train_df.write.mode("overwrite").parquet(train_path)

    print(f"💾 Saving TEST to: {test_path}")
    test_df.write.mode("overwrite").parquet(test_path)
    
def save_as_parquet(df):
    output_path = PATHS["parquet"]
    print(f"💾 Saving to: {output_path}")

    df.write.mode("overwrite").parquet(output_path)


def run_etl():
    spark = create_spark_session()

    df = load_raw_data(spark)

    df = select_relevant_columns(df)

    df = clean_data(df)

    df = filter_active_users_items(df)   # 🔥 NEW (important)

    # Performance
    df = df.repartition(200)
    df.cache()
    df.count()

    #train_df, test_df = user_based_split(df)

    #save_train_test(train_df, test_df)
    save_as_parquet(df)
    print("✅ ETL Completed")
    spark.stop()


if __name__ == "__main__":
    run_etl()