#core processing stage before graph model. reviewerId and asin string but GraphX needs numbers.
# encode_ids.py

from pyspark.sql.functions import monotonically_increasing_id
from config import PATHS
from etl import create_spark_session


def encode_ids(spark, df):
    """
    Encode user and item IDs to numeric values
    """

    # -------------------------
    # Encode Users
    # -------------------------
    users = df.select("reviewerID").distinct() \
              .withColumn("user_id", monotonically_increasing_id())

    df = df.join(users, on="reviewerID", how="inner")

    # -------------------------
    # Encode Items
    # -------------------------
    items = df.select("asin").distinct() \
              .withColumn("item_id", monotonically_increasing_id())

    df = df.join(items, on="asin", how="inner")

    # -------------------------
    # Select final columns
    # -------------------------
    df = df.select("user_id", "item_id", "overall", "unixReviewTime")

    return df


def run_encoding():
    spark = create_spark_session()

    input_path = PATHS["parquet"] + "/5core"
    output_path = PATHS["parquet"] + "/encoded"

    print(f"📥 Loading: {input_path}")
    df = spark.read.parquet(input_path)

    df_encoded = encode_ids(spark, df)

    print(f"💾 Saving: {output_path}")
    df_encoded.write.mode("overwrite").parquet(output_path)

    print("✅ ID Encoding Completed")

    spark.stop()


if __name__ == "__main__":
    run_encoding()