from pyspark.sql.functions import col, lit, max as spark_max, sum as spark_sum
from pyspark.sql.window import Window
from config import PATHS
from etl import create_spark_session


def build_graph(spark, df):
    """
    Build bipartite graph (vertices + edges)
    """

    # -------------------------
    # Step 1: Get max user_id
    # -------------------------
    max_user_id = df.agg(spark_max("user_id")).collect()[0][0]
    print(f"Max user_id: {max_user_id}")

    # -------------------------
    # Step 2: Shift item IDs
    # -------------------------
    df = df.withColumn(
        "item_id_shifted",
        col("item_id") + max_user_id + 1
    )

    # -------------------------
    # Step 3: Create forward edges (user → item)
    # -------------------------
    edges = df.select(
        col("user_id").alias("src"),
        col("item_id_shifted").alias("dst"),
        col("overall").cast("double").alias("weight")
    )

    # Normalize rating (0–1)
    edges = edges.withColumn("weight", col("weight") / 5.0)

    # -------------------------
    # Step 4: Add reverse edges (item → user)
    # -------------------------
    reverse_edges = edges.select(
        col("dst").alias("src"),
        col("src").alias("dst"),
        col("weight")
    )

    edges = edges.union(reverse_edges)

    # -------------------------
    # Step 5: Normalize (row stochastic)
    # -------------------------
    window = Window.partitionBy("src")

    edges = edges.withColumn(
        "weight",
        col("weight") / spark_sum("weight").over(window)
    )

    # -------------------------
    # Step 6: Create vertices
    # -------------------------
    users = df.select(
        col("user_id").alias("id")
    ).distinct().withColumn("type", lit("user"))

    items = df.select(
        col("item_id_shifted").alias("id")
    ).distinct().withColumn("type", lit("item"))

    vertices = users.union(items)

    # -------------------------
    # Cache (important)
    # -------------------------
    vertices = vertices.cache()
    edges = edges.cache()

    print(f"Vertices: {vertices.count()}")
    print(f"Edges: {edges.count()}")

    return vertices, edges


def run_graph_builder():
    spark = create_spark_session()

    input_path = PATHS["parquet"] + "/encoded"
    vertex_path = PATHS["graph"] + "/vertices"
    edge_path = PATHS["graph"] + "/edges"

    print(f"📥 Loading: {input_path}")
    df = spark.read.parquet(input_path)

    vertices, edges = build_graph(spark, df)


    print("💾 Saving vertices...: ", vertices.count())
    vertices.coalesce(10).write.mode("overwrite").parquet(vertex_path)

    print("💾 Saving edges...: ", edges.count())
    edges.write.mode("overwrite").parquet(edge_path)

    print("✅ Graph Construction Completed")
    spark.stop()


if __name__ == "__main__":
    run_graph_builder()