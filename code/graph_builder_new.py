from pyspark.sql.functions import col, lit, max as spark_max, sum as spark_sum
from pyspark.sql.window import Window
from config import PATHS
from etl import create_spark_session

from pyspark.sql.functions import col, collect_list
from pyspark.sql.functions import max as spark_max

from pyspark.sql.functions import (
    col, collect_list, struct,
    log1p, max as spark_max
)
from pyspark.sql.window import Window
from pyspark.sql.functions import sum as spark_sum

def build_graph_node2vec(df, alpha=0.3, beta=0.2):
    """
    Build weighted adjacency list for Node2Vec
    """

    # -------------------------
    # Step 1: Get max user_id
    # -------------------------
    max_user_id = df.select(spark_max("user_id")).first()[0]
    print(f"Max user_id: {max_user_id}")

    # -------------------------
    # Step 2: Shift item IDs
    # -------------------------
    df = df.withColumn(
        "item_id_shifted",
        col("item_id") + max_user_id + 1
    )

    # -------------------------
    # Step 3: Compute edge weight
    # -------------------------
    df = df.withColumn(
        "rating_norm",
        col("rating") / 5.0
    ).withColumn(
        "vote_norm",
        log1p(col("vote"))
    ).withColumn(
        "verified_num",
        col("verified").cast("int")
    )

    df = df.withColumn(
        "weight",
        col("rating_norm") *
        (1 + alpha * col("vote_norm")) *
        (1 + beta * col("verified_num"))
    )

    # -------------------------
    # Step 4: Create edges
    # -------------------------
    edges = df.select(
        col("user_id").alias("src"),
        col("item_id_shifted").alias("dst"),
        col("weight")
    )

    # -------------------------
    # Step 5: Make bidirectional
    # -------------------------
    reverse_edges = edges.select(
        col("dst").alias("src"),
        col("src").alias("dst"),
        col("weight")
    )

    edges = edges.union(reverse_edges)

    # -------------------------
    # Step 6: Normalize weights
    # -------------------------
    w = Window.partitionBy("src")

    edges = edges.withColumn(
        "weight",
        col("weight") / (spark_sum("weight").over(w) + 1e-9)
    )

    # -------------------------
    # Step 7: Build adjacency list
    # -------------------------
    adj = edges.groupBy("src").agg(
        collect_list(struct("dst", "weight")).alias("neighbors")
    )

    print(f"Adjacency nodes: {adj.count()}")

    return adj, max_user_id


import random

import random

import random

def generate_walks(adj_df, num_walks=5, walk_length=10):

    # ⚠️ collect adjacency (still heavy, but OK for small graph)
    adj_data = adj_df.select("src", "neighbors").collect()

    # Build dictionary
    adj_dict = {
        row["src"]: row["neighbors"]
        for row in adj_data
    }

    # Get list of nodes (IMPORTANT FIX)
    nodes = list(adj_dict.keys())

    walks = []

    for node in nodes[:10000]:   # ✅ limit safely
        for _ in range(num_walks):

            walk = [str(node)]
            current = node

            for _ in range(walk_length - 1):

                neighbors = adj_dict.get(current, [])

                if not neighbors:
                    break

                # ✅ Weighted sampling
                next_nodes = [n["dst"] for n in neighbors]
                weights = [n["weight"] for n in neighbors]

                current = random.choices(next_nodes, weights=weights, k=1)[0]

                walk.append(str(current))

            walks.append(walk)

    return walks
    

from pyspark.sql.functions import col, lit, sum as spark_sum, max as spark_max
from pyspark.sql.window import Window

def build_graph(spark, df):

    spark.sparkContext.setCheckpointDir(PATHS["checkpoint"])

    # -------------------------
    # Step 1: Get max user_id (safe)
    # -------------------------
    max_user_id = df.select(spark_max("user_id")).first()[0]
    print(f"Max user_id: {max_user_id}")

    # -------------------------
    # Step 2: Shift item IDs
    # -------------------------
    df = df.withColumn(
        "item_id_shifted",
        col("item_id") + max_user_id + 1
    )

    df = df.repartition(200)  # 🔥 important for joins/window ops

    # -------------------------
    # Step 3: Forward edges
    # -------------------------
    edges = df.select(
        col("user_id").alias("src"),
        col("item_id_shifted").alias("dst"),
        (col("rating") / 5.0).alias("weight")   # assuming renamed column
    )

    # -------------------------
    # Step 4: Reverse edges
    # -------------------------
    reverse_edges = edges.select(
        col("dst").alias("src"),
        col("src").alias("dst"),
        col("weight")
    )

    edges = edges.union(reverse_edges).distinct()  # 🔥 avoid duplicates

    # -------------------------
    # Step 5: Normalize weights
    # -------------------------
    degree_df = edges.groupBy("src").agg(
        spark_sum("weight").alias("total_weight")
    )

    edges = edges.join(degree_df, "src")

    edges = edges.withColumn(
        "weight",
        col("weight") / col("total_weight")
    ).drop("total_weight")

    # -------------------------
    # Step 6: Vertices
    # -------------------------
    users = df.select(
        col("user_id").alias("id")
    ).distinct().withColumn("type", lit("user"))

    items = df.select(
        col("item_id_shifted").alias("id")
    ).distinct().withColumn("type", lit("item"))

    vertices = users.union(items)

    # -------------------------
    # Step 7: Checkpoint + Cache
    # -------------------------
    edges = edges.checkpoint().cache()
    vertices = vertices.checkpoint().cache()

    # Materialize
    print(f"Vertices: {vertices.count()}")
    print(f"Edges: {edges.count()}")

    return vertices, edges


def run_graph_builder(spark, df, max_user_id, mode='train'):

    vertex_path = PATHS["graph"] + "/vertices/" + mode
    edge_path = PATHS["graph"] + "/edges/" + mode

    vertices, edges = build_graph(spark, df)

    print("💾 Saving vertices...")
    vertices.write.mode("overwrite").parquet(vertex_path)

    print("💾 Saving edges...")
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