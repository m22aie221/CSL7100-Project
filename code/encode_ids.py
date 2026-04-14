from pyspark.sql.window import Window
from pyspark.sql.functions import dense_rank
from config import PATHS
from etl import create_spark_session

from pyspark.sql.functions import monotonically_increasing_id, broadcast


def encode_ids(spark, df):

    df = df.repartition(32, "reviewerID")

    # Users
    users = df.select("reviewerID").distinct() \
        .withColumn("user_id", monotonically_increasing_id())

    # Items
    items = df.select("asin").distinct() \
        .withColumn("item_id", monotonically_increasing_id())

    # Join (broadcast for speed)
    df = df.join(broadcast(users), on="reviewerID")
    df = df.join(broadcast(items), on="asin")

    return df.select("user_id", "item_id", "overall", "unixReviewTime")


def encode_ids_old(spark, df):

    # Encode Users
    user_window = Window.orderBy("reviewerID")

    users = df.select("reviewerID").distinct() \
        .withColumn("user_id", dense_rank().over(user_window))

    df = df.join(users, on="reviewerID", how="inner")

    # Encode Items
    item_window = Window.orderBy("asin")

    items = df.select("asin").distinct() \
        .withColumn("item_id", dense_rank().over(item_window))

    df = df.join(items, on="asin", how="inner")

    df = df.select("user_id", "item_id", "overall", "unixReviewTime")

    return df


def run_encoding():
    spark = create_spark_session()

    input_path = PATHS["parquet"] + "/5core"
    output_path = PATHS["parquet"] + "/encoded"

    df = spark.read.parquet(input_path)

    df_encoded = encode_ids(spark, df)

    #df_encoded.write.mode("overwrite").parquet(output_path)

    if df_encoded.count() > 0:
        df_encoded.coalesce(10).write.mode("overwrite").parquet(PATHS["parquet"] + "/encoded")
        print("✅ Encoded saved")
    else:
        print("❌ Encoding produced empty dataframe")
    
    print("✅ ID Encoding Completed")
    spark.stop()


if __name__ == "__main__":
    run_encoding()