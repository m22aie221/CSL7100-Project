# etl.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from config import PATHS, USE_HDFS


def create_spark_session(app_name="RecommenderGraph-ETL"):
    """
    Create Spark session based on storage type
    """
    builder = SparkSession.builder.appName(app_name)

    if USE_HDFS:
        # HDFS mode
        builder = builder.config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
        print("🚀 Running in HDFS mode")
    else:
        # Local mode
        builder = builder.config("spark.hadoop.fs.defaultFS", "file:///").config("spark.driver.memory", "4g").config("spark.executor.memory", "4g")
        print("💻 Running in LOCAL mode")

    spark = builder.getOrCreate()
    return spark


def load_raw_data(spark):
    """
    Load raw JSON dataset
    """
    input_path = PATHS["raw"]
    print(f"📥 Loading data from: {input_path}")

    df = spark.read.json(input_path)
    return df


def select_relevant_columns(df):
    """
    Select required columns
    """
    required_cols = ["reviewerID", "asin", "overall", "unixReviewTime"]

    df = df.select([col(c) for c in required_cols])
    return df

from pyspark.sql.functions import col

def clean_data(df):
    df = df.dropna(subset=["reviewerID", "asin", "overall"])
    print("✅ Removed rows with null user/item/rating")
    return df

def save_as_parquet(df):
    """
    Save to Parquet
    """
    output_path = PATHS["parquet"]
    print(f"💾 Saving to: {output_path}")

    df.write.mode("overwrite").parquet(output_path)


def run_etl():
    """
    Full ETL pipeline
    """
    spark = create_spark_session()

    df = load_raw_data(spark)
    df = select_relevant_columns(df)
    df = clean_data(df)

    save_as_parquet(df)

    print("✅ ETL Completed")

    spark.stop()


if __name__ == "__main__":
    run_etl()