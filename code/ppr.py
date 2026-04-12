from pyspark.sql.functions import col, lit, sum as spark_sum, abs as spark_abs


def personalized_pagerank(spark, vertices, edges, source_id, alpha=0.15, max_iter=20, tol=1e-6):
    """
    Optimized Personalized PageRank
    """

    # -------------------------
    # Step 1: Initialize ranks
    # -------------------------
    ranks = vertices.select("id").withColumn(
        "rank",
        (col("id") == source_id).cast("double")
    )

    for i in range(max_iter):
        print(f"🔁 Iteration {i+1}")

        ranks = ranks.cache()

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
        new_ranks = vertices.select("id").join(
            agg,
            vertices.id == agg.dst,
            "left"
        ).select(
            vertices.id,
            col("rank")
        ).fillna(0.0)

        # -------------------------
        # Step 5: Apply damping
        # -------------------------
        new_ranks = new_ranks.withColumn(
            "rank",
            (1 - alpha) * col("rank")
        )

        # -------------------------
        # Step 6: Teleport to source
        # -------------------------
        new_ranks = new_ranks.withColumn(
            "rank",
            col("rank") + alpha * (col("id") == source_id).cast("double")
        )

        # -------------------------
        # Step 7: Convergence check
        # -------------------------
        diff = new_ranks.join(
            ranks.withColumnRenamed("rank", "prev_rank"),
            "id"
        ).select(
            spark_abs(col("rank") - col("prev_rank")).alias("diff")
        ).agg({"diff": "max"}).collect()[0][0]

        print(f"Max diff: {diff}")

        ranks = new_ranks

        if diff < tol:
            print("✅ Converged")
            break

    return ranks