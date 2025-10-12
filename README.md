Overview
========
Data Engineering Pipeline — Datafest Hackathon 2025 (Team XYZ)

![Architecture Diagram](architecture_diagram/diagram.png)

This project implements a data engineering pipeline that automates the flow of raw data from Amazon S3 into Amazon Redshift, transforms it using dbt (Data Build Tool), and orchestrates the entire process with Apache Airflow.

Architecture
=================================

- s3-to-redshift.py: Copies CSV files from S3 to Redshift staging tables using S3ToRedshiftOperator.

- dbt_transformations.py: Triggers dbt transformations after data load completion.

- dbt: Cleans, models, and materializes data into analytics-ready schemas.

Data Source
=================================
To read on how the data used for this project was obtained, click here [How Data Was Generated](HOW_DATA_WAS_GENERATED.md)

Tech Stack
=================================

- Orchestration: Apache Airflow
- Transformation: dbt + Redshift
- Storage: AWS S3
- Data Warehouse: Amazon Redshift Serverless
- Environment: WSL (Ubuntu)

Project Contents
================

This project contains some important files and folders:

- dags: This folder contains the Python files for Airflow DAGs:
    `s3-to-redshift.py`: This DAG automates the process of moving raw data from Amazon s3 bucket into Amazon redshift, and into staging schema for later transformation.
- Dockerfile: This file contains a versioned Astro Runtime Docker image that provides a differentiated Airflow experience. If you want to execute other commands or overrides at runtime, specify them here.
- include: This folder contains any additional files that you want to include as part of your project. It is empty by default.
- requirements.txt: This was used to install important requireents like amazon and Postgres providers. Without then, airflow can not connect to these services
- dbt_env: This folder contains the virtual environment for dbt. It isolates all dbt related dependencies from the main Airflow environment, ensuring clean version management and reproducible builds. Please note that it is system-specific. It was Created using:  
 ```
  python3 -m venv dbt_env.

```
Activated with:
```
  source dbt_env/bin/activate
```
Deactivated with:
```
  deactivate
```

- dbt_transformations_datathon: This is the part responsible for transforming and modeling data in Amazon Redshift after Airflow has ingested to staging schema.
- HOW_DATA_WAS_GENERATED.md: This is a file that explains how the data used for this report was generated.
- Power BI Report - Team XYZ.pdf: This is a file that contains our powerbi dashboard report.
- phc_aiapp: Streamlit-powered PHC AI Dashboard that connects to Amazon Redshift to visualize, forecast, and interact with health facility, patient, inventory, and disease data.


Running the Pipeline
===========================

Start Airflow on your local machine
```
 'astro dev start'.

```
When project image has been buily, open the browser to the Airflow UI at http://localhost:8080/

- Run s3_to_redshift dag from the airflow UI, it then loads S3 data into Redshift staging schema.
- Run dbt_transformations dag to activate dbt environment and executes dbt run.

Exploratory Data Analysis and Report
========================================
- To explore our findings based on the dataset, here is a link to our powerbi interactive dashboard. click [Team XYZ PHC report](https://app.powerbi.com/view?r=eyJrIjoiNzQ4Yjg2ZDEtMTMyOS00NzdkLTg4ZTUtNGNkYWIwMzhjNzQ4IiwidCI6ImQwMjZkZjkxLTg4MTYtNDIyNS04YmIyLWMzODJhOGUxZWYwYyJ9).
- To read an explicit report on our finding, click here [Data_Analysis_Report](Data_analysis_report.pdf)
- To interact with our AI model click [AI Model](https://datathon2025-42evmbyj3noewwrz3bcj55.streamlit.app/)


Notes
=================================

- This pipeline is idempotent, staging tables in redshift are truncated before reloading.
- Airflow connection to s3 bucket and redshift were done via the airflow UI and not hardcoded.
