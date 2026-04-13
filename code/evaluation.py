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