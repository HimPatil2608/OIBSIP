# Data Analytics Internship Portfolio

This repository contains the projects and tasks completed during my Data Analytics Internship at **Oasis Infobyte**.  
The portfolio demonstrates a progression of skills ranging from fundamental data cleaning and exploratory data analysis to complex machine learning modeling and natural language processing.

---

## 📋 Table of Contents

- Task 1: Retail Sales Analysis  
- Task 2: Customer Segmentation  
- Task 3: Data Cleaning (NYC Airbnb)  
- Task 4: Sentiment Analysis  
- Task 5: House Price Prediction  
- Task 6: Wine Quality Prediction  
- Task 7: Credit Card Fraud Detection  
- Task 8: Google Play Store App Analysis  
- Task 9: Autocomplete & Autocorrect System  

---

## 🛠 Technologies Used

**Languages:**  
- Python  

**Libraries:**  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- NLTK  
- TextBlob  
- Plotly  

---

## Task 1: Retail Sales Analysis

**File:** Task 1.pdf  

### 📝 Overview
Performed a comprehensive exploratory data analysis (EDA) on a retail sales dataset to identify trends in customer purchasing behavior.

### 🔍 Key Analysis
- **Data Cleaning:** Handled missing values and standardized date formats.  
- **Descriptive Statistics:** Calculated mean, median, mode, and standard deviation for transactional data.  
- **Time Series Analysis:** Analyzed daily and monthly sales trends to identify peak periods.  
- **Customer & Product Analysis:** Identified top-spending customers and best-selling product categories.  

### 📊 Results
- Visualized sales trends showing seasonality and daily fluctuations.  
- Determined that *"Clothing"* was a dominant product category.  
- Generated insights on sales distribution across different demographics.  

---

## Task 2: Customer Segmentation

**File:** Task 2.pdf  

### 📝 Overview
Used unsupervised learning techniques to segment customers based on their purchasing habits and demographics.

### 🔍 Methodology
- **Preprocessing:** Handled missing values and engineered features like `Total_Spending` and `Total_Transactions`.  
- **Clustering:** Applied K-Means Clustering. Used the Elbow Method to determine the optimal number of clusters ($k=4$).  
- **Dimensionality Reduction:** Used PCA (Principal Component Analysis) to visualize clusters in 2D space.  

### 📊 Insights
- **Cluster 0 (High Income & Spend):** Premium customers; target with loyalty programs.  
- **Cluster 1 (Moderate):** Growth segment; target with combo offers.  
- **Cluster 2 (Low Income/Spend):** Price-sensitive; target with discounts.  
- **Cluster 3 (High Income, Low Spend):** Upsell opportunities; consider luxury campaigns.  

---

## Task 3: Data Cleaning (NYC Airbnb)

**File:** Task 3.pdf  

### 📝 Overview
Focused on data integrity and preparation using the NYC Airbnb 2019 dataset.

### 🔍 Key Steps
- **Missing Value Imputation:** Filled missing values in `reviews_per_month` and `host_name`.  
- **Data Type Correction:** Converted dates to datetime objects and ensured numerical consistency.  
- **Outlier Removal:** Used the IQR (Interquartile Range) method to remove extreme price outliers.  
- **Consistency Checks:** Validated `availability_365` and `minimum_nights` logic.  

### 📊 Outcome
- Produced a clean, consistent dataset (`AB_NYC_2019_cleaned.csv`) ready for reliable analysis.  
- Visualized price distributions before and after outlier removal.  

---

## Task 4: Sentiment Analysis

**File:** Task 4.pdf  

### 📝 Overview
Built a text classification model to analyze public sentiment from Twitter data.

### 🔍 Methodology
- **Text Preprocessing:** Cleaned text data and mapped labels (Negative, Neutral, Positive).  
- **Feature Extraction:** Used TF-IDF Vectorizer to convert text into numerical features.  
- **Modeling:** Trained a LinearSVC (Support Vector Classifier).  

### 📊 Results
- Achieved an Accuracy of ~85.8%.  
- Generated a Classification Report and Confusion Matrix to evaluate precision and recall across all sentiment classes.  

---

## Task 5: House Price Prediction

**File:** Task 5.pdf  

### 📝 Overview
Developed a regression model to predict housing prices based on features like area, bedrooms, and amenities.

### 🔍 Methodology
- **Exploratory Analysis:** Correlation heatmaps to identify key price drivers.  
- **Preprocessing:** One-hot encoding for categorical variables.  
- **Modeling:** Trained a Linear Regression model.  

### 📊 Results
- **R² Score:** ~0.65 (The model explains 65% of the variance in pricing).  
- Visualized Actual vs. Predicted prices and Residual plots to check for homoscedasticity.  

---

## Task 6: Wine Quality Prediction

**File:** Task 6.pdf  

### 📝 Overview
Compared multiple classification algorithms to predict the quality of wine based on physicochemical tests.

### 🔍 Methodology
- **Models Tested:** Random Forest, SGD Classifier, SVC (Support Vector Classifier).  
- **Evaluation:** Used Confusion Matrices and Classification Reports.  

### 📊 Results
- Random Forest outperformed other models with an accuracy of ~72%.  
- Identified key chemical features contributing to higher wine quality.  

---

## Task 7: Credit Card Fraud Detection

**File:** Task 7.pdf  

### 📝 Overview
Built a robust fraud detection system handling highly imbalanced data.

### 🔍 Methodology
- **Data Split:** Stratified train-test split to maintain class ratios.  
- **Scaling:** Applied RobustScaler to handle outliers.  
- **Models:** Logistic Regression, Decision Tree, and MLP Classifier (Neural Network).  
- **Balancing:** Used `class_weight='balanced'` to handle the rarity of fraud cases.  

### 📊 Results
**ROC-AUC Scores:**  
- Logistic Regression: ~0.947  
- Neural Network: ~0.893  
- Decision Tree: ~0.862  

The models successfully identified fraud cases with high recall, crucial for financial security.

---

## Task 8: Google Play Store App Analysis

**File:** Task 8.pdf  

### 📝 Overview
Analyzed the Android app market to understand app trends, pricing strategies, and user sentiment.

### 🔍 Key Analysis
- **Data Cleaning:** Converted sizes (M/k) to bytes, cleaned price and install columns.  
- **Category Analysis:** Visualized the most popular app categories (Family, Game, Tools).  
- **Rating vs. Installs:** Investigated the correlation between app popularity and user ratings.  
- **Sentiment Analysis:** Merged user reviews to calculate sentiment polarity for apps.  

### 📊 Insights
- "Family" and "Game" categories dominate the market.  
- Paid apps tend to have a lower volume of installs but often higher engagement.  

---

## Task 9: Autocomplete & Autocorrect System

**File:** Task 9.pdf  

### 📝 Overview
Implemented an NLP-based system for text autocompletion and correction using probabilistic language models.

### 🔍 Methodology
- **Corpus:** Used NLTK's Brown and Gutenberg corpora.  
- **Tokenization:** Normalized and tokenized text using RegEx.  
- **N-Grams:** Built Unigram, Bigram, and Trigram models to predict the next word.  
- **Autocorrect:** Implemented Levenshtein distance (edit distance) to suggest corrections for misspelled words (Norvig's method).  

### 📊 Results
- Demonstrated effective autocomplete suggestions based on previous word context.  
- Achieved high efficiency in correcting common typos like *"teh"* → *"the"*.  

---

## 🙏 Acknowledgments

I would like to thank **Oasis Infobyte** for providing the opportunity to work on these diverse datasets and challenges.  
This internship significantly enhanced my practical skills in Data Science and Machine Learning.

---

## 📜 Citations & Datasets

- **Retail Sales / House Pricing / Wine Quality / Credit Card Fraud / Airbnb:**  
  Datasets typically sourced from Kaggle or UCI Machine Learning Repository.  

- **NLP Corpora:**  
  Provided by the NLTK Project.
