import os

# -----------------------------------
# 🔧 User Config
# -----------------------------------

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/home/suvendu/mlbd/project1/CSL7100-Project")

USE_HDFS = False  # os.getenv("USE_HDFS", "false").lower() == "true"

HDFS_BASE = "hdfs://localhost:9000/project"

# -----------------------------------
# 📁 Path Resolver
# -----------------------------------

def get_base_path():
    if USE_HDFS:
        return HDFS_BASE
    else:
        return f"file://{PROJECT_ROOT}"

BASE_PATH = get_base_path()

PATHS = {
    "raw": f"{BASE_PATH}/raw",
    "parquet": f"{BASE_PATH}/parquet",
    "graph": f"{BASE_PATH}/parquet/graph",
    "checkpoint": f"{BASE_PATH}/checkpoints",
    "spark_temp": f"{BASE_PATH}/spark-temp"
}

# -----------------------------------
# 🧠 Debug Info
# -----------------------------------

def print_config():
    print("🔧 CONFIGURATION")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"USE_HDFS: {USE_HDFS}")
    print(f"BASE_PATH: {BASE_PATH}")
    print("PATHS:")
    for k, v in PATHS.items():
        print(f"  {k}: {v}")