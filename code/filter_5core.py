# filter_5core.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from config import PATHS
from etl import create_spark_session


def filter_k_core_old(df, k=5, max_iter=10):
    """
    Iterative k-core filtering
    Keeps users and items with >= k interactions
    """

    prev_count = -1

    for i in range(max_iter):
        print(f"\n🔁 Iteration {i+1}")

        # Filter users
        user_counts = df.groupBy("reviewerID").agg(count("*").alias("user_count"))
        df = df.join(user_counts, "reviewerID") \
               .filter(col("user_count") >= k) \
               .drop("user_count")

        # Filter items
        item_counts = df.groupBy("asin").agg(count("*").alias("item_count"))
        df = df.join(item_counts, "asin") \
               .filter(col("item_count") >= k) \
               .drop("item_count")

        # Check convergence
        curr_count = df.count()
        print(f"Records after iteration {i+1}: {curr_count}")

        if curr_count == prev_count:
            print("✅ Converged")
            break

        prev_count = curr_count

    return df

def filter_k_core_7iter(df, k=5, max_iter=10):

    df = df.persist()

    for i in range(max_iter):
        print(f"\n🔁 Iteration {i+1}")

        user_counts = df.groupBy("reviewerID").count()
        df = df.join(user_counts, "reviewerID") \
               .filter(col("count") >= k) \
               .drop("count")

        item_counts = df.groupBy("asin").count()
        df = df.join(item_counts, "asin") \
               .filter(col("count") >= k) \
               .drop("count")

        df = df.persist()

    return df

def filter_k_core(spark, df, k=5, max_iter=10):

    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

    prev_count = -1

    for i in range(max_iter):
        print(f"\n🔁 Iteration {i+1}")

        user_counts = df.groupBy("reviewerID").count()
        df = df.join(user_counts, "reviewerID") \
               .filter(col("count") >= k) \
               .drop("count")

        item_counts = df.groupBy("asin").count()
        df = df.join(item_counts, "asin") \
               .filter(col("count") >= k) \
               .drop("count")

        # 🔥 BREAK LINEAGE
        df = df.checkpoint()

        curr_count = df.limit(1).count()  # lightweight trigger

        if curr_count == prev_count:
            print("✅ Converged")
            break

        prev_count = curr_count

    return df

def run_5core():
    """
    Full pipeline for 5-core filtering
    """
    spark = create_spark_session()

    input_path = PATHS["parquet"]
    output_path = PATHS["parquet"] + "/5core"

    print(f"📥 Loading data from: {input_path}")
    df = spark.read.parquet(input_path)
    df = df.sample(0.2)

    print(f"Initial records: {df.count()}")

    df_filtered = filter_k_core(spark, df, k=5)

    print(f"💾 Saving filtered data to: {output_path}")
    df_filtered.write.mode("overwrite").parquet(output_path)

    print("✅ 5-core filtering completed")

    spark.stop()


if __name__ == "__main__":
    run_5core()