from pyspark.sql.window import Window
from pyspark.sql.functions import dense_rank
from config import PATHS
from etl import create_spark_session
from pyspark.ml.feature import StringIndexer

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



def encode_ids_new(spark, df):

    print("🔢 Encoding user_id and item_id...")

    # Encode Users
    user_indexer = StringIndexer(
        inputCol="user_id",
        outputCol="user_idx",
        handleInvalid="skip"
    )

    # Encode Items
    item_indexer = StringIndexer(
        inputCol="item_id",
        outputCol="item_idx",
        handleInvalid="skip"
    )

    df = user_indexer.fit(df).transform(df)
    df = item_indexer.fit(df).transform(df)

    # Cast to int (important for ML models)
    df = df.withColumn("user_id", df["user_idx"].cast("int")) \
           .withColumn("item_id", df["item_idx"].cast("int"))

    # Drop temporary columns
    df = df.drop("user_idx", "item_idx")

    return df

def run_encoding():
    spark = create_spark_session()

    input_path = PATHS["parquet"] + "/5core"
    output_path = PATHS["parquet"] + "/encoded"

    df = spark.read.parquet(input_path)

    df_encoded = encode_ids(spark, df)

    df_encoded.write.mode("overwrite").parquet(output_path)

    print("✅ ID Encoding Completed")
    spark.stop()


if __name__ == "__main__":
    run_encoding()