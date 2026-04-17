from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
from config import PATHS, USE_HDFS
from pyspark.sql.functions import col, rand, row_number
from pyspark.sql.window import Window

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rand, col

def user_based_split(df, test_ratio=0.2):

    print("🔀 Performing user-based split")

    # ✅ Use user_id (NOT reviewerID)
    window = Window.partitionBy("user_id").orderBy(rand(seed=42))

    df = df.withColumn("row_num", row_number().over(window))

    counts = df.groupBy("user_id").count()

    df = df.join(counts, "user_id")

    df = df.withColumn(
        "split",
        (col("row_num") / col("count") > (1 - test_ratio))
    )

    train_df = df.filter(~col("split")).drop("row_num", "count", "split")
    test_df = df.filter(col("split")).drop("row_num", "count", "split")

    print(f"Train: {train_df.count()}, Test: {test_df.count()}")

    return train_df, test_df

    
def user_based_split_old(df, test_ratio=0.2):

    window = Window.partitionBy("reviewerID").orderBy(rand())

    df = df.withColumn("row_num", row_number().over(window))

    counts = df.groupBy("reviewerID").count()

    df = df.join(counts, "reviewerID")

    df = df.withColumn(
        "split",
        (col("row_num") / col("count") > (1 - test_ratio))
    )

    train_df = df.filter(~col("split")).drop("row_num", "count", "split")
    test_df = df.filter(col("split")).drop("row_num", "count", "split")

    return train_df, test_df

def create_spark_session(app_name="RecommenderGraph-ETL"):
    builder = SparkSession.builder.appName(app_name)

    if USE_HDFS:
        builder = (
            builder
            .master("local[6]")  # use 6 cores, leave 2 for system
            .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
            .config("spark.driver.memory", "3g")
            .config("spark.executor.memory", "3g")
            .config("spark.sql.shuffle.partitions", "48")
        )
        print("🚀 Running in HDFS mode")
    else:
        builder = (
            builder.config("spark.hadoop.fs.defaultFS", "file:///")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
        )
        print("💻 Running in LOCAL mode")

    return builder.getOrCreate()


def load_raw_data(spark):
    input_path = PATHS["raw"]
    print(f"📥 Loading data from: {input_path}")

    schema = StructType([
        StructField("reviewerID", StringType(), True),
        StructField("asin", StringType(), True),
        StructField("overall", DoubleType(), True),
        StructField("unixReviewTime", LongType(), True),
    ])

    df = spark.read.schema(schema).json(input_path)
    df = df.sample(0.05) #TBD
    return df


def select_relevant_columns(df):
    return df.select("reviewerID", "asin", "overall", "unixReviewTime")


def clean_data(df):
    df = df.dropna(subset=["reviewerID", "asin", "overall"])
    print("✅ Removed nulls")
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


def inspect_raw_data(spark):
    input_path = PATHS["raw"]
    print(f"📥 Loading full data from: {input_path}")

    # Load without schema → infers full schema
    df = spark.read.json(input_path)

    print("\n📌 Full Schema:")
    df.printSchema()

    print("\n📌 Sample Records (3 rows):")
    df.show(3, truncate=False)

    return df

def run_etl():
    spark = create_spark_session()

    inspect_raw_data(spark)
    
    df = load_raw_data(spark)
    df = select_relevant_columns(df)
    df = clean_data(df)

    # Performance improvements
    df = df.repartition(200)
    df.cache()
    df.count()

    #train_df, test_df = user_based_split(df)
    # Save 
    #save_train_test(train_df, test_df)
    save_as_parquet(df)

    print("✅ ETL Completed")
    spark.stop()


if __name__ == "__main__":
    run_etl()