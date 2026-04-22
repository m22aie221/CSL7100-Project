from pyspark.sql.functions import col
from config import PATHS
from etl import create_spark_session

from pyspark.sql.functions import col

from pyspark.sql.functions import col

def estimate_optimal_k(df, spark, k_values=[2,3,4,5], sample_frac=0.2):
    print("\n🔍 Estimating optimal k...")

    spark.sparkContext.setCheckpointDir(PATHS["checkpoint"])

    # ✅ Reduce sample size (important)
    df_sample = df.sample(fraction=sample_frac, seed=42).cache()
    df_sample.count()  # materialize

    results = []

    for k in k_values:
        print(f"\nTesting k={k}")

        df_k = df_sample

        for i in range(3):  # 🔥 reduce iterations
            print(f"  Iteration {i+1}")

            # User filtering
            user_counts = df_k.groupBy("user_id").count()
            valid_users = user_counts.filter(col("count") >= k).select("user_id")
            df_k = df_k.join(valid_users, "user_id", "semi")

            # Item filtering
            item_counts = df_k.groupBy("item_id").count()
            valid_items = item_counts.filter(col("count") >= k).select("item_id")
            df_k = df_k.join(valid_items, "item_id", "semi")

            # ✅ Break lineage (VERY IMPORTANT)
            df_k = df_k.checkpoint(eager=True)

        users = df_k.select("user_id").distinct().count()
        items = df_k.select("item_id").distinct().count()
        edges = df_k.count()

        print(f"k={k} → users={users}, items={items}, edges={edges}")

        results.append((k, users, items, edges))

    # ✅ safer heuristic
    best_k = max(results, key=lambda x: x[3])[0]

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

        # ✅ Filter users (distributed)
        user_counts = df.groupBy("user_id").count()
        valid_users = user_counts.filter(col("count") >= k).select("user_id")

        df = df.join(valid_users, "user_id")

        # ✅ Filter items (distributed)
        item_counts = df.groupBy("item_id").count()
        valid_items = item_counts.filter(col("count") >= k).select("item_id")

        df = df.join(valid_items, "item_id")

        # ✅ Break lineage
        df = df.checkpoint().cache()

    return df
    


def run_5core():
    spark = create_spark_session()

    input_path = PATHS["parquet"]
    output_path = PATHS["parquet"] + "/5core"

    print(f"📥 Loading data from: {input_path}")
    df = spark.read.parquet(input_path)

    print(f"Initial records: {df.count()}")

    k = estimate_optimal_k(df, spark)
    #k=4 # parameter tuned
    df_filtered = filter_k_core_optimized(spark, df, k=k)
    #df_filtered = filter_k_core(spark, df, k=3)

    print(f"💾 Saving filtered data to: {output_path}")
    df_filtered.write.mode("overwrite").parquet(output_path)

    print("✅ 5-core filtering completed")
    spark.stop()


if __name__ == "__main__":
    run_5core()