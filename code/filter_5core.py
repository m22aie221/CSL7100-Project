from pyspark.sql.functions import col
from config import PATHS
from etl import create_spark_session

def estimate_optimal_k(df, spark, k_values=[2,3,4,5], sample_frac=0.3):
    print("\n🔍 Estimating optimal k...")

    # Sample for speed
    df_sample = df.sample(fraction=sample_frac, seed=42)

    results = []

    for k in k_values:
        print(f"\nTesting k={k}")

        df_k = df_sample

        for _ in range(5):  # few iterations (approximate)
            user_counts = df_k.groupBy("reviewerID").count()
            df_k = df_k.join(user_counts, "reviewerID") \
                       .filter(col("count") >= k) \
                       .drop("count")

            item_counts = df_k.groupBy("asin").count()
            df_k = df_k.join(item_counts, "asin") \
                       .filter(col("count") >= k) \
                       .drop("count")

        users = df_k.select("reviewerID").distinct().count()
        items = df_k.select("asin").distinct().count()
        edges = df_k.count()

        print(f"k={k} → users={users}, items={items}, edges={edges}")

        results.append((k, users, items, edges))

    # Heuristic: choose largest graph with reasonable filtering
    best_k = max(results, key=lambda x: (x[1] * x[2], x[3]))[0]

    print(f"\n✅ Selected optimal k = {best_k}")
    return best_k
    
# Improve Spark Performance 
def filter_k_core_optimized(spark, df, k=5, max_iter=10):

    spark.sparkContext.setCheckpointDir(PATHS["checkpoint"])

    df = df.repartition(100).cache()

    prev_count = -1

    for i in range(max_iter):
        print(f"\n🔁 Iteration {i+1}")

        curr_count = df.count()
        print(f"Records before filtering: {curr_count}")

        if curr_count == prev_count:
            print("✅ Converged")
            break

        prev_count = curr_count

        # ✅ Filter users (NO JOIN)
        user_counts = df.groupBy("reviewerID").count()
        valid_users = [row["reviewerID"] for row in user_counts.filter(col("count") >= k).collect()]

        df = df.filter(col("reviewerID").isin(valid_users))

        # ✅ Filter items (NO JOIN)
        item_counts = df.groupBy("asin").count()
        valid_items = [row["asin"] for row in item_counts.filter(col("count") >= k).collect()]

        df = df.filter(col("asin").isin(valid_items))

        df = df.checkpoint().cache()

    return df
    
# need to remove (Remove Expensive Joins)
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
        df.groupBy("reviewerID").count().describe().show()
        df.groupBy("asin").count().describe().show()
        print("Users:", df.select("reviewerID").distinct().count())
        print("Items:", df.select("asin").distinct().count())
    return df


def run_5core():
    spark = create_spark_session()

    input_path = PATHS["parquet"]
    output_path = PATHS["parquet"] + "/5core"

    print(f"📥 Loading data from: {input_path}")
    df = spark.read.parquet(input_path)

    print(f"Initial records: {df.count()}")

    #k = estimate_optimal_k(df, spark)
    k=2 # parameter tuned
    df_filtered = filter_k_core_optimized(spark, df, k=k)
    #df_filtered = filter_k_core(spark, df, k=3)

    print(f"💾 Saving filtered data to: {output_path}")
    df_filtered.write.mode("overwrite").parquet(output_path)

    print("✅ 5-core filtering completed")
    spark.stop()


if __name__ == "__main__":
    run_5core()