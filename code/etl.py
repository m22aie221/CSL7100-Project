from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
from config import PATHS, USE_HDFS


def create_spark_session(app_name="RecommenderGraph-ETL"):
    builder = SparkSession.builder.appName(app_name)

    if USE_HDFS:
        builder = builder.config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
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


def save_as_parquet(df):
    output_path = PATHS["parquet"]
    print(f"💾 Saving to: {output_path}")

    df.write.mode("overwrite").parquet(output_path)


def run_etl():
    spark = create_spark_session()

    df = load_raw_data(spark)
    df = select_relevant_columns(df)
    df = clean_data(df)

    # Performance improvements
    df = df.repartition(200)
    df.cache()
    df.count()

    save_as_parquet(df)

    print("✅ ETL Completed")
    spark.stop()


if __name__ == "__main__":
    run_etl()