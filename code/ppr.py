from pyspark.sql.functions import col, lit, sum as spark_sum


def personalized_pagerank_old(spark, vertices, edges, source_id, alpha=0.15, max_iter=10):
    """
    Compute Personalized PageRank using DataFrame API
    """

    # -------------------------
    # Step 1: Initialize ranks
    # -------------------------
    ranks = vertices.withColumn(
        "rank",
        lit(1.0 if False else 0.0)  # placeholder
    )

    # Set source node rank = 1
    ranks = ranks.withColumn(
        "rank",
        col("id").cast("long")
    ).withColumn(
        "rank",
        (col("id") == source_id).cast("double")
    )

    # -------------------------
    # Iterative updates
    # -------------------------
    for i in range(max_iter):
        print(f"Iteration {i+1}")

        # Join edges with ranks
        contribs = edges.join(
            ranks,
            edges.src == ranks.id,
            "inner"
        ).select(
            col("dst"),
            (col("rank") * col("weight")).alias("contrib")
        )

        # Aggregate contributions
        new_ranks = contribs.groupBy("dst").agg(
            spark_sum("contrib").alias("rank")
        )

        # Apply damping
        new_ranks = new_ranks.withColumn(
            "rank",
            (1 - alpha) * col("rank") + alpha * lit(0.0)
        )

        # Add teleport to source node
        new_ranks = new_ranks.withColumn(
            "rank",
            col("rank") + alpha * (col("dst") == source_id).cast("double")
        )

        # Rename for next iteration
        ranks = new_ranks.withColumnRenamed("dst", "id")

    return ranks

from pyspark.sql.functions import col, lit, sum as spark_sum


def personalized_pagerank(spark, vertices, edges, source_id, alpha=0.15, max_iter=10):

    # -------------------------
    # Step 1: Initialize ranks
    # -------------------------
    ranks = vertices.select("id").withColumn(
        "rank",
        (col("id") == source_id).cast("double")
    )

    for i in range(max_iter):
        print(f"Iteration {i+1}")

        # -------------------------
        # Step 2: Contributions
        # -------------------------
        contribs = edges.join(
            ranks,
            edges.src == ranks.id,
            "inner"
        ).select(
            col("dst"),
            (col("rank") * col("weight")).alias("contrib")
        )

        # -------------------------
        # Step 3: Aggregate
        # -------------------------
        agg = contribs.groupBy("dst").agg(
            spark_sum("contrib").alias("rank")
        )

        # -------------------------
        # Step 4: Keep ALL nodes
        # -------------------------
        new_ranks = vertices.select(col("id")).join(
            agg,
            vertices.id == agg.dst,
            "left"
        ).select(
            vertices.id,
            col("rank")
        )

        # Fill missing with 0
        new_ranks = new_ranks.fillna(0.0)

        # -------------------------
        # Step 5: Apply damping
        # -------------------------
        new_ranks = new_ranks.withColumn(
            "rank",
            (1 - alpha) * col("rank")
        )

        # -------------------------
        # Step 6: Add teleport
        # -------------------------
        new_ranks = new_ranks.withColumn(
            "rank",
            col("rank") + alpha * (col("id") == source_id).cast("double")
        )

        ranks = new_ranks

    return ranks