# 🚀 RecommenderGraph – Project Description

RecommenderGraph is a distributed recommendation system built using Apache Spark and GraphX. It models user-item interactions as a bipartite graph and applies Personalized PageRank (PPR) to generate scalable and personalized recommendations.

The system focuses on efficient large-scale data processing using Spark, memory-aware design, and optimized storage formats such as Parquet. It separates code and data storage to ensure flexibility and portability across different environments.

---

# 📁 Project Folder Structure

```
CSL7100-Project/
│
├── code/                      # Core implementation
│   ├── config.py              # Path configuration using environment variables
│   ├── etl.py                 # Data ingestion and preprocessing
│   ├── filter_5core.py        # 5-core filtering logic
│   ├── encode_ids.py          # User/item ID encoding
│   ├── graph_build.py         # Graph construction (GraphX)
│   ├── ppr_model.py           # Personalized PageRank logic
│   ├── recommend.py           # Generate Top-N recommendations
│   └── utils.py               # Helper functions
│
├── notebooks/                 # Jupyter notebooks
│   ├── preprocessing.ipynb
│   └── recommendation_demo.ipynb
│
├── output/                    # Final outputs
│   └── recommendations.json
│
├── reports/                   # Documentation
│   └── final_report.pdf
│
├── requirements.txt
├── README.md
└── .bashrc / .env             # Environment variables (PROJECT_ROOT)

------------------------------------------------------------

External Data Directory (Not part of repo):

D:\IITJ\Course\sem7\BigData\Project
│
├── raw/                       # Original dataset (JSON)
├── parquet/                   # Processed data (Parquet)
├── hdfs/                      # Simulated HDFS (optional)
└── spark-temp/                # Spark temporary files
```

---

# ⚙️ Configuration

The project uses an environment variable to handle system-specific paths:

```
PROJECT_ROOT=/mnt/d/IITJ/Course/sem7/BigData/Project
```

All data paths are dynamically resolved in `config.py`, ensuring compatibility across different machines.

---

# 🧠 Summary

The project separates code and data storage to support large datasets, improve reproducibility, and enable seamless collaboration across different system configurations.



nano ~/.bashrc
export PROJECT_ROOT=/mnt/d/IITJ/Course/sem7/BigData/Project
