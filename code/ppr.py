from pyspark.sql.functions import col, lit, sum as spark_sum, abs as spark_abs


from pyspark.sql.functions import col, lit, sum as spark_sum, abs as spark_abs, broadcast
from pyspark import StorageLevel


def personalized_pagerank_optimized(
    spark,
    vertices,
    edges,
    source_id,
    alpha=0.15,
    max_iter=20,
    tol=1e-6,
    top_n=10
):
    """
    Optimized Personalized PageRank for recommender systems
    """

    print("🚀 Starting Personalized PageRank")

    # -------------------------
    # Step 0: Make graph bidirectional
    # -------------------------
    print("🔄 Making graph bidirectional")

    edges_rev = edges.select(
        col("dst").alias("src"),
        col("src").alias("dst"),
        col("weight")
    )

    edges = edges.union(edges_rev).persist(StorageLevel.MEMORY_AND_DISK)

    #  CRITICAL: Normalize weights
    edge_sums = edges.groupBy("src").agg(
        spark_sum("weight").alias("total_weight")
    )
    
    edges = edges.join(edge_sums, "src") \
                 .withColumn("weight", col("weight") / col("total_weight")) \
                 .drop("total_weight")
    # -------------------------
    # Step 1: Cache vertices
    # -------------------------
    vertices = vertices.select("id", "type").persist(StorageLevel.MEMORY_AND_DISK)

    # -------------------------
    # Step 2: Initialize ranks
    # -------------------------
    ranks = vertices.select("id").withColumn(
        "rank",
        (col("id") == source_id).cast("double")
    ).persist(StorageLevel.MEMORY_AND_DISK)

    
    print(f"🎯 Source user: {source_id}")

    # -------------------------
    # Iterations
    # -------------------------
    for i in range(max_iter):
        print(f"\n🔁 Iteration {i+1}")

        # -------------------------
        # Step 3: Contributions (Broadcast for speed)
        # -------------------------
        contribs = edges.join(
            broadcast(ranks),
            edges.src == ranks.id,
            "inner"
        ).select(
            col("dst").alias("id"),
            (col("rank") * col("weight")).alias("contrib")
        )

        # -------------------------
        # Step 4: Aggregate contributions
        # -------------------------
        agg = contribs.groupBy("id").agg(
            spark_sum("contrib").alias("rank")
        )

        # -------------------------
        # Step 5: Keep ALL nodes (left join)
        # -------------------------
        new_ranks = vertices.select("id").join(
            agg,
            "id",
            "left"
        ).fillna(0.0)

        # -------------------------
        # Step 6: Apply damping
        # -------------------------
        new_ranks = new_ranks.withColumn(
            "rank",
            (1 - alpha) * col("rank")
        )

        # -------------------------
        # Step 7: Teleport to source
        # -------------------------
        new_ranks = new_ranks.withColumn(
            "rank",
            col("rank") + alpha * (col("id") == source_id).cast("double")
        )

        new_ranks = new_ranks.persist(StorageLevel.MEMORY_AND_DISK)
        if i % 3 == 0:
            new_ranks = new_ranks.checkpoint()
       
        # -------------------------
        # Step 8: Convergence check (efficient)
        # -------------------------
        diff = new_ranks.join(
            ranks.withColumnRenamed("rank", "prev_rank"),
            "id"
        ).select(
            spark_abs(col("rank") - col("prev_rank")).alias("diff")
        )

        max_diff = diff.agg(spark_sum("diff")).first()[0]

        print(f"Max diff: {max_diff}")

        # Prepare next iteration
        ranks.unpersist()
        contribs.unpersist()
        agg.unpersist()
        ranks = new_ranks
        
        if max_diff < tol:
            print("✅ Converged")
            break

    # -------------------------
    # Step 9: Extract recommendations (items only)
    # -------------------------
    print("\n🎯 Extracting Top-N recommendations")

    recommendations = ranks.join(vertices, "id")

    top_items = recommendations.filter(
        col("type") == "item"
    ).orderBy(col("rank").desc()).limit(top_n)

    return ranks, top_items

    

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