from pyspark.sql.functions import col, size, array_intersect, collect_list

def evaluate_precision_recall(recommendations, ground_truth, k=5):
    """
    recommendations: (user_id, item_id, rank)
    ground_truth: (user_id, actual_items)
    """

    # -------------------------
    # Top-K recommendations per user
    # -------------------------
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number

    window = Window.partitionBy("user_id").orderBy(col("rank").desc())

    top_k = recommendations.withColumn(
        "rank_pos", row_number().over(window)
    ).filter(col("rank_pos") <= k)

    top_k = top_k.groupBy("user_id").agg(
        collect_list("item_id").alias("pred_items")
    )

    # -------------------------
    # Join with ground truth
    # -------------------------
    eval_df = top_k.join(ground_truth, "user_id")

    # -------------------------
    # Compute metrics
    # -------------------------
    eval_df = eval_df.withColumn(
        "intersection",
        array_intersect(col("pred_items"), col("actual_items"))
    )

    eval_df = eval_df.withColumn(
        "precision",
        size(col("intersection")) / k
    ).withColumn(
        "recall",
        size(col("intersection")) / size(col("actual_items"))
    )

    # -------------------------
    # Aggregate
    # -------------------------
    result = eval_df.agg(
        {"precision": "avg", "recall": "avg"}
    ).collect()[0]

    precision = result["avg(precision)"]
    recall = result["avg(recall)"]

    print(f"\n✅ Precision@{k}: {precision}")
    print(f"✅ Recall@{k}: {recall}")

    return precision, recall



from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

def average_precision(pred, actual, k):
    if not actual:
        return 0.0

    score = 0.0
    hits = 0

    for i, p in enumerate(pred[:k]):
        if p in actual:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(actual), k)

ap_udf = udf(lambda pred, actual: average_precision(pred, actual, 5), DoubleType())


def compute_map(recommendations, ground_truth, k=5):
    eval_df = recommendations.join(ground_truth, "user_id")

    eval_df = eval_df.withColumn(
        "ap",
        ap_udf(col("pred_items"), col("actual_items"))
    )

    map_k = eval_df.agg({"ap": "avg"}).collect()[0][0]

    print(f"✅ MAP@{k}: {map_k}")

    return map_k    


from pyspark.sql.functions import col, countDistinct

def evaluate_precision_recall_safe(recommendations, ground_truth, k=5):

    # Top-K already assumed
    recs = recommendations.select("user_id", "item_id")

    # Explode ground truth
    gt = ground_truth.selectExpr("user_id", "explode(actual_items) as item_id")

    # Intersection
    hits = recs.join(gt, ["user_id", "item_id"])

    # Count hits per user
    hits_count = hits.groupBy("user_id").agg(
        countDistinct("item_id").alias("hits")
    )

    # Count actual items
    gt_count = gt.groupBy("user_id").agg(
        countDistinct("item_id").alias("actual_count")
    )

    # Combine
    eval_df = hits_count.join(gt_count, "user_id")

    eval_df = eval_df.withColumn(
        "precision", col("hits") / k
    ).withColumn(
        "recall", col("hits") / col("actual_count")
    )

    result = eval_df.agg(
        {"precision": "avg", "recall": "avg"}
    ).collect()[0]

    return result["avg(precision)"], result["avg(recall)"]  


from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, when, sum as spark_sum

def compute_map_safe(recommendations, ground_truth, k=5):

    # Rank recommendations
    window = Window.partitionBy("user_id").orderBy(col("rank").desc())

    recs = recommendations.withColumn(
        "rank_pos", row_number().over(window)
    ).filter(col("rank_pos") <= k)

    # Explode ground truth
    gt = ground_truth.selectExpr("user_id", "explode(actual_items) as item_id")

    # Mark hits
    recs = recs.join(
        gt.withColumn("is_hit", col("item_id")),
        ["user_id", "item_id"],
        "left"
    ).withColumn(
        "hit", when(col("is_hit").isNotNull(), 1).otherwise(0)
    )

    # Cumulative hits
    window2 = Window.partitionBy("user_id").orderBy("rank_pos")

    recs = recs.withColumn(
        "cum_hits", spark_sum("hit").over(window2)
    )

    # Precision at each position
    recs = recs.withColumn(
        "precision_at_k",
        col("cum_hits") / col("rank_pos")
    )

    # Only keep hits
    recs = recs.filter(col("hit") == 1)

    # AP per user
    ap = recs.groupBy("user_id").agg(
        spark_sum("precision_at_k").alias("ap")
    )

    # Normalize by min(|actual|, k)
    gt_count = ground_truth.selectExpr(
        "user_id", "size(actual_items) as actual_count"
    )

    ap = ap.join(gt_count, "user_id")

    ap = ap.withColumn(
        "ap",
        col("ap") / when(col("actual_count") < k, col("actual_count")).otherwise(k)
    )

    # MAP
    map_k = ap.agg({"ap": "avg"}).collect()[0][0]

    return map_k    