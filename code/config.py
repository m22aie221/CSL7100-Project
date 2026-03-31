import os
from pyspark.sql import SparkSession

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/default")

PATHS = {
    "raw": f"{PROJECT_ROOT}/raw",
    "parquet": f"{PROJECT_ROOT}/parquet",
    "hdfs": f"{PROJECT_ROOT}/hdfs",
    "spark_temp": f"{PROJECT_ROOT}/spark-temp"
}

import os

PROJECT_ROOT = os.getenv("PROJECT_ROOT")




