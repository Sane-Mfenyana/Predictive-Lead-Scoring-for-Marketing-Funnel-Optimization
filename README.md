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

## Step 2: Data Discovery & Quality Check (SQL)

**Objective:** Perform initial exploratory analysis in BigQuery to understand dataset structure, assess data quality, and identify key patterns and predictors.

**Actions Taken:**

1.  **Volume & Basic Statistics:** Verified dataset size and basic client demographics.
    ```sql
    SELECT 
      COUNT(*) AS total_rows,
      COUNT(DISTINCT age) AS unique_ages,
      AVG(age) AS avg_age,
      MIN(age) AS min_age,
      MAX(age) AS max_age
    FROM `bank_marketing.bank_additional_full`;
    ```
    *   **Result:** 41,188 rows. Client age range: 17-98 years, average ~40.

2.  **Target Variable Analysis:** Calculated the baseline marketing conversion rate.
    ```sql
    SELECT 
      y,
      COUNT(*) AS count,
      ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
    FROM `bank_marketing.bank_additional_full`
    GROUP BY y;
    ```
    *   **Result:** Overall conversion rate (`y = true`) is **11.27%**. This class imbalance is typical for marketing funnel data and sets the baseline for model performance.

3.  **Predictor Analysis - Housing Loans:** Tested the hypothesis that existing housing loans influence subscription rates.
    ```sql
    SELECT 
      housing,
      COUNT(*) AS total_clients,
      COUNTIF(y = true) AS subscribed_yes,
      ROUND(COUNTIF(y = true) * 100.0 / COUNT(*), 2) AS conversion_rate_percent
    FROM `bank_marketing.bank_additional_full`
    GROUP BY housing
    ORDER BY conversion_rate_percent DESC;
    ```
    *   **Finding:** Clients with a housing loan (`yes`) convert at **11.62%**, slightly higher than those without (`no`, 10.88%). The effect is minor.

4.  **Predictor Analysis - Previous Outcome:** Identified the single most powerful predictor.
    ```sql
    SELECT 
      poutcome,
      COUNT(*) AS total_clients,
      COUNTIF(y = true) AS subscribed_yes,
      ROUND(COUNTIF(y = true) * 100.0 / COUNT(*), 2) AS conversion_rate_percent
    FROM `bank_marketing.bank_additional_full`
    GROUP BY poutcome
    ORDER BY conversion_rate_percent DESC;
    ```
    *   **Key Insight:** Clients with a `'success'` in a previous campaign have a **65.11% conversion rate**, which is **7.4x higher** than new leads (`'nonexistent'`, 8.83%). This is the #1 lever for marketing efficiency.

5.  **Data Quality Check:** Verified the integrity of the critical `poutcome` column.
    ```sql
    SELECT COUNT(*) AS rows_with_null_poutcome
    FROM `bank_marketing.bank_additional_full`
    WHERE poutcome IS NULL;
    ```
    *   **Result:** 0 rows with NULL values. Data quality is excellent for this column.

6.  **Engagement Analysis:** Compared call duration between converted and non-converted leads.
    ```sql
    SELECT 
      y AS subscribed,
      ROUND(AVG(duration) / 60, 1) AS avg_duration_minutes
    FROM `bank_marketing.bank_additional_full`
    GROUP BY y;
    ```
    *   **Finding:** Successful calls last **9.2 minutes** on average, compared to **3.7 minutes** for unsuccessful calls. This indicates a significant engagement threshold for conversion.

**Tools:** BigQuery (SQL).

**Key Learnings:**
*   The dataset is clean and ready for analysis, with no NULLs in critical columns.
*   The `poutcome` column is an exceptionally strong predictor, highlighting the high value of existing customers.
*   Significant engagement (call duration) is strongly correlated with conversion.

---

## Step 3: Business Context & Funnel Definition

**Objective:** Translate the dataset and initial findings into a structured business framework. Define the marketing funnel stages, key performance indicators (KPIs), and the analytical approach for the project.

**Actions Taken:**

1.  **Funnel Stage Definition:** Mapped the customer journey to three measurable stages:
    *   **Stage 1 - Contacted:** Lead was contacted (`contact` is not null).
    *   **Stage 2 - Engaged:** Call duration lasted **5 minutes or more** (`duration >= 300` seconds).
    *   **Stage 3 - Converted:** Lead subscribed to the term deposit (`y = true`).

2.  **Funnel Quantification:** Executed SQL to calculate volumes and conversion rates between each stage.
    ```sql
    WITH funnel AS (
      SELECT
        COUNT(*) AS total_contacted,
        COUNTIF(duration >= 300) AS engaged,
        COUNTIF(y = true) AS converted
      FROM `bank_marketing.bank_additional_full`
    )
    SELECT 
      total_contacted,
      engaged,
      converted,
      ROUND(engaged * 100.0 / total_contacted, 2) AS contact_to_engage_rate,
      ROUND(converted * 100.0 / engaged, 2) AS engage_to_convert_rate,
      ROUND(converted * 100.0 / total_contacted, 2) AS overall_conversion_rate
    FROM funnel;
    ```
    *   **Results:** 41,188 contacted → 11,250 engaged (27.31%) → 4,640 converted (41.24%). Overall conversion rate: 11.27%.

3.  **KPI Establishment:** Defined four core metrics for ongoing performance tracking:
    *   Engagement Rate, Engaged Conversion Rate, Overall Conversion Rate, and Average Conversion Duration.

4.  **Modeling Strategy Formulation:** Adopted a dual-model approach to serve both analytical and operational needs:
    *   **Diagnostic Model:** Includes `duration` to profile successful conversions and understand key drivers.
    *   **Operational Model:** Excludes `duration` (unknown pre-call) to provide a true lead score for prioritizing marketing calls.

**Tools:** BigQuery (SQL), Business Analysis.

**Key Insight:** The largest funnel drop-off occurs at the engagement stage (only 27% of calls last >5 minutes). Improving this is the primary lever for increasing conversions.

---
