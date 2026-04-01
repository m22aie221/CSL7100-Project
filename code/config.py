# config.py

import os

# -----------------------------------
# 🔧 User Config
# -----------------------------------

# Set this via environment OR default
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/IITJ/Course/sem7/BigData/Project")

# Toggle storage type
USE_HDFS = False #os.getenv("USE_HDFS", "false").lower() == "true"

# HDFS base (only used if USE_HDFS=True)
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
    "hdfs": f"{BASE_PATH}/hdfs",
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



