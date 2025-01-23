!["Ineuron"](https://ineuron.ai/images/ineuron-logo.png)


# Customer Personality Analysis Ineuron Intelligence Private Limited Project

## Overview
Customer Personality Analysis is a machine learning project that aims to segment customers into distinct groups based on their demographic and behavioral attributes. These insights can be used for targeted marketing, personalized recommendations, and customer engagement strategies.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Objective](#objective)
- [Methodology](#methodology)
- [Clustering Insights](#clustering-insights)
- [Machine Learning Models](#machine-learning-models)
- [Tools and Libraries Used](#tools-and-libraries-used)
- [Results](#results)
- [Business Applications](#business-applications)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

---

## Dataset
The project uses the `marketing_campaign.csv` dataset containing **2240 rows** and **29 columns**.  
Key columns include:
- **Demographics:** Year_Birth, Education, Marital_Status, Income  
- **Purchase Behavior:** Amount spent on products (Wines, Fruits, Meat, etc.), number of purchases through various channels.  
- **Customer Engagement:** Recency, Campaign acceptance, Complaints, etc.

---

## Objective
1. **Segment Customers**: Group customers into clusters based on their behaviors and demographics.  
2. **Understand Cluster Characteristics**: Identify the unique attributes of each segment.  
3. **Build Predictive Models**: Use classification algorithms to predict cluster memberships.  
4. **Actionable Insights**: Provide recommendations for targeted marketing campaigns.

---

## Methodology
### 1. Data Preparation
- Handled missing values and removed redundant features.  
- Engineered new features like `Spent`, `Age`, `Family_Size`, `Is_Parent`, etc., to improve clustering.  
- Simplified categorical variables for better interpretability.

### 2. Dimensionality Reduction
- Applied **Principal Component Analysis (PCA)** to reduce dimensions while retaining ~60% variance.

### 3. Clustering
- Used **Agglomerative Clustering** to segment customers into 4 clusters.  
- Evaluated cluster quality using **silhouette scores**.

### 4. Classification Models
- Trained and evaluated models to predict cluster memberships:
  - Decision Tree
  - Random Forest
  - XGBoost (best-performing model with **95% accuracy**).

---

## Clustering Insights
The analysis identified **4 customer clusters**:

- **Cluster 0**:  
  - Younger parents, smaller families.  
  - Moderate spending.  

- **Cluster 1**:  
  - Older parents, larger families (2–4 members), teenagers present.  

- **Cluster 2**:  
  - Non-parents, couples or singles.  
  - High-income and high-spending group.  

- **Cluster 3**:  
  - Older parents, very large families, often with teenagers.  

---

## Machine Learning Models
- **Decision Tree Classifier**: 93% accuracy.  
- **Random Forest Classifier**: 94% accuracy.  
- **XGBoost Classifier**: 95% accuracy (best model).  

The **XGBoost model** was serialized using `pickle` for deployment.

---

## Tools and Libraries Used
- **Languages:** Python  
- **Libraries:**  
  - Data Manipulation: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`, `plotly`  
  - Machine Learning: `scikit-learn`, `XGBoost`, `Yellowbrick`  
  - Clustering: `KMeans`, `AgglomerativeClustering`, `PCA`

---

## Results
- **Dimensionality Reduction:** Achieved with PCA (3 components).  
- **Clustering:** Four meaningful customer groups identified.  
- **Model Performance:** XGBoost achieved the best accuracy of **95%**.  
- **Output:** Clustered dataset saved as `clustered_customer.csv` and model saved as `model.pkl`.

---

## Business Applications
1. **Targeted Marketing**: Create customized campaigns for each customer cluster.  
2. **Product Recommendations**: Offer relevant products based on spending behaviors.  
3. **Customer Retention**: Improve loyalty programs for high-value clusters.  
4. **Resource Optimization**: Focus efforts on high-revenue-generating segments.

---

## How to Run
### Prerequisites
- Python 3.8 or above.
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
