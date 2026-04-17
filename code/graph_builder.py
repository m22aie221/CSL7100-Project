from pyspark.sql.functions import col, lit, max as spark_max, sum as spark_sum
from pyspark.sql.window import Window
from config import PATHS
from etl import create_spark_session

from pyspark.sql.functions import col, collect_list
from pyspark.sql.functions import max as spark_max

def build_graph_node2vec(df):
    """
    Build adjacency list for Node2Vec (random walks)
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
    # Step 3: Create edges (NO WEIGHT)
    # -------------------------
    edges = df.select(
        col("user_id").alias("src"),
        col("item_id_shifted").alias("dst")
    )

    # -------------------------
    # Step 4: Make bidirectional
    # -------------------------
    reverse_edges = edges.select(
        col("dst").alias("src"),
        col("src").alias("dst")
    )

    edges = edges.union(reverse_edges)

    # -------------------------
    # Step 5: Build adjacency list
    # -------------------------
    adj = edges.groupBy("src").agg(
        collect_list("dst").alias("neighbors")
    )

    print(f"Adjacency nodes: {adj.count()}")

    return adj, max_user_id


import random

def generate_walks(adj_df, num_walks=5, walk_length=10):

    adj_dict = {
        row["src"]: row["neighbors"]
        for row in adj_df.collect()
    }

    walks = []

    for node in adj_dict.keys():
        for _ in range(num_walks):

            walk = [str(node)]
            current = node

            for _ in range(walk_length - 1):

                neighbors = adj_dict.get(current, [])

                if not neighbors:
                    break

                current = random.choice(neighbors)
                walk.append(str(current))

            walks.append(walk)

    return walks
    

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

from pyspark.sql.functions import col, lit, sum as spark_sum, max as spark_max


def build_graph(spark, df):

    # Cache input
    df = df.cache()
    df.count()

    # Max user_id
    max_user_id = df.selectExpr("max(user_id) as max_id").first()["max_id"]
    print(f"Max user_id: {max_user_id}")

    # Shift item IDs
    df = df.withColumn(
        "item_id_shifted",
        col("item_id") + max_user_id + 1
    )

    # Forward edges
    edges = df.select(
        col("user_id").alias("src"),
        col("item_id_shifted").alias("dst"),
        (col("overall") / 5.0).cast("double").alias("weight")
    )

    # Reverse edges
    reverse_edges = edges.select(
        col("dst").alias("src"),
        col("src").alias("dst"),
        col("weight")
    )

    edges = edges.unionByName(reverse_edges)

    # Repartition for performance
    edges = edges.repartition(48, "src")

    # Normalize (WITHOUT window)
    norm = edges.groupBy("src").agg(spark_sum("weight").alias("total_weight"))

    edges = edges.join(norm, on="src") \
                 .withColumn("weight", col("weight") / col("total_weight")) \
                 .drop("total_weight")

    # Vertices
    users = df.select(col("user_id").alias("id")).distinct().withColumn("type", lit("user"))
    items = df.select(col("item_id_shifted").alias("id")).distinct().withColumn("type", lit("item"))

    vertices = users.unionByName(items)

    # Cache
    vertices = vertices.cache()
    edges = edges.cache()

    print(f"Vertices: {vertices.count()}")
    print(f"Edges: {edges.count()}")

    return vertices, edges

def run_graph_builder(spark, df, max_user_id, mode='train'):

    vertex_path = PATHS["graph"] + "/vertices/" + mode
    edge_path = PATHS["graph"] + "/edges/" + mode

    vertices, edges = build_graph(spark, df)


    print("💾 Saving vertices...: ", vertices.count())
    vertices.coalesce(10).write.mode("overwrite").parquet(vertex_path)

    print("💾 Saving edges...: ", edges.count())
    edges.write.mode("overwrite").parquet(edge_path)

    print(f"✅ Graph Construction Completed for {mode}")

    print("✅ Graph Construction Completed")
    #spark.stop()


if __name__ == "__main__":
    spark = create_spark_session()
    df = spark.read.parquet(PATHS["parquet"] + "/encoded")

    max_user_id = df.agg(spark_max("user_id")).collect()[0][0]

    run_graph_builder(spark, df, max_user_id, mode='train')
    #run_graph_builder()