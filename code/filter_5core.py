from pyspark.sql.functions import col
from config import PATHS
from etl import create_spark_session


def filter_k_core(spark, df, k=5, max_iter=10):

    spark.sparkContext.setCheckpointDir(PATHS["checkpoint"])

    df = df.repartition(100)

    prev_count = -1

    for i in range(max_iter):
        print(f"\n🔁 Iteration {i+1}")

        df.cache()

        # Trigger once
        curr_count = df.count()
        print(f"Records before filtering: {curr_count}")

        if curr_count == prev_count:
            print("✅ Converged")
            break

        prev_count = curr_count

        # Filter users
        user_counts = df.groupBy("reviewerID").count()
        df = df.join(user_counts, "reviewerID") \
               .filter(col("count") >= k) \
               .drop("count")

        # Filter items
        item_counts = df.groupBy("asin").count()
        df = df.join(item_counts, "asin") \
               .filter(col("count") >= k) \
               .drop("count")

        # Break lineage
        df = df.checkpoint()

    return df


def run_5core():
    spark = create_spark_session()

    input_path = PATHS["parquet"]
    output_path = PATHS["parquet"] + "/5core"

    print(f"📥 Loading data from: {input_path}")
    df = spark.read.parquet(input_path)

    print(f"Initial records: {df.count()}")

    df_filtered = filter_k_core(spark, df, k=5)

    print(f"💾 Saving filtered data to: {output_path}")
    df_filtered.write.mode("overwrite").parquet(output_path)

    print("✅ 5-core filtering completed")
    spark.stop()


if __name__ == "__main__":
    run_5core()