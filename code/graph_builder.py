# graph_builder.py

from pyspark.sql.functions import lit, col, max as spark_max
from config import PATHS
from etl import create_spark_session


def build_graph_old(spark, df):
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
    df = df.withColumn("item_id_shifted", col("item_id") + max_user_id + 1)

    # -------------------------
    # Step 3: Create Edges
    # -------------------------
    edges = df.select(
        col("user_id").alias("src"),
        col("item_id_shifted").alias("dst"),
        col("overall").alias("weight")
    )   
    
    # -------------------------
    # Step 4: Create Vertices
    # -------------------------
    users = df.select(col("user_id").alias("id")).distinct() \
              .withColumn("type", lit("user"))

    items = df.select(col("item_id_shifted").alias("id")).distinct() \
              .withColumn("type", lit("item"))

    vertices = users.union(items)

    return vertices, edges


from pyspark.sql.functions import col, lit, max as spark_max, sum as spark_sum
from pyspark.sql.window import Window


from pyspark.sql.functions import col, lit, max as spark_max


def build_graph(spark, df):
    """
    Build bipartite graph (vertices + edges)
    SAFE version (no zero-weight issues)
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
    # Step 3: Create Edges
    # -------------------------
    edges = df.select(
        col("user_id").alias("src"),
        col("item_id_shifted").alias("dst"),
        col("overall").alias("weight")
    )

    # -------------------------
    # Step 4: SAFE Normalization
    # -------------------------
    edges = edges.withColumn(
        "weight",
        col("weight").cast("double") / 5.0
    )

    # -------------------------
    # OPTIONAL: Advanced normalization (USE ONLY AFTER DEBUG)
    # -------------------------
    """
    from pyspark.sql.window import Window
    from pyspark.sql.functions import sum as spark_sum

    window = Window.partitionBy("src")

    edges = edges.withColumn(
        "weight",
        col("weight").cast("double") / spark_sum("weight").over(window)
    )
    """

    # -------------------------
    # Step 5: Create Vertices
    # -------------------------
    users = df.select(
        col("user_id").alias("id")
    ).distinct().withColumn("type", lit("user"))

    items = df.select(
        col("item_id_shifted").alias("id")
    ).distinct().withColumn("type", lit("item"))

    vertices = users.union(items)

    # -------------------------
    # Debug prints (IMPORTANT)
    # -------------------------
    print(f"Vertices: {vertices.count()}")
    print(f"Edges: {edges.count()}")

    return vertices, edges

def run_graph_builder():
    spark = create_spark_session()

    input_path = PATHS["parquet"] + "/encoded"
    vertex_path = PATHS["parquet"] + "/graph/vertices"
    edge_path = PATHS["parquet"] + "/graph/edges"

    print(f"📥 Loading: {input_path}")
    df = spark.read.parquet(input_path)

    vertices, edges = build_graph(spark, df)

    print("💾 Saving vertices...")
    vertices.write.mode("overwrite").parquet(vertex_path)

    print("💾 Saving edges...")
    edges.write.mode("overwrite").parquet(edge_path)

    print("✅ Graph Construction Completed")

    spark.stop()

if __name__ == "__main__":
    run_graph_builder()