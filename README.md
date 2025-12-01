# Predictive Lead Scoring for Marketing Funnel Optimization

## Project Description
This end-to-end analytics project addresses a core challenge in FinTech marketing: inefficient lead prioritization. By analyzing an enriched customer interaction dataset, I developed a machine learning model to predict conversion likelihood and built an interactive Tableau dashboard for funnel visualization. The project demonstrates the full commercial analytics lifecycle—from SQL data wrangling and exploratory analysis to predictive modeling and business intelligence—showcasing how data science directly informs strategic marketing spend and customer acquisition strategy.

## Business Problem
A financial institution's telemarketing campaign aims to sell term deposit products. The marketing team needs to:
1.  **Understand** the customer journey and identify key drop-off points in the marketing funnel.
2.  **Predict** which leads are most likely to convert, enabling efficient prioritization and resource allocation for sales agents.
3.  **Monitor** campaign performance and ROI in real-time to optimize future marketing spend.

A manual, reactive approach leads to inefficient spend and missed opportunities. This project provides a data-driven solution through predictive lead scoring and executive dashboards.

## Repository Structure
```
├── /sql/ # BigQuery SQL scripts for data preparation
├── /notebooks/ # Jupyter Notebook for EDA & modelling
├── /dashboard/ # Tableau workbook and related assets
├── /docs/ # Project summary and presentation
└── README.md # This file
```
---

## Step 1: Project Setup & Data Ingestion
**Objective:** Establish the foundational project environment and ingest the primary dataset into a cloud data warehouse.

**Actions Taken:**
1.  **Repository Initialization:** Created a public GitHub repository with the title "Predictive Lead Scoring for Marketing Funnel Optimization". The repository was initialized with a `Python` .gitignore file and an `MIT License`.
2.  **Cloud Infrastructure:** Set up a Google Cloud Platform project and enabled the BigQuery API.
3.  **Data Pipeline:** Created a BigQuery dataset named `bank_marketing` and uploaded the enriched `bank-additional.csv` file as the primary table, `bank_additional_full`.

**Tools:** GitHub, Google Cloud Platform Console, BigQuery.
