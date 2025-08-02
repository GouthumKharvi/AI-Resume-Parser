# AI-Resume-Parser
An AI-powered job recommendation system that uses transformer-based sentence embeddings to semantically match candidate resumes with job postings, enhancing recruitment efficiency and candidate-job fit beyond traditional keyword-based search.

Absolutely! Here’s a fully expanded, **pin-to-pin detailed README** with everything — including AI techniques, model details, Streamlit web app info, how to run, and requirements. This version leaves nothing out:

---

````markdown
# AI-Powered Job Recommendation System

---

## Project Overview  
This project implements an end-to-end AI-driven job recommendation system that semantically matches candidate resumes with job postings. Leveraging transformer-based sentence embeddings, it enhances traditional keyword matching to improve HR efficiency and candidate selection.

---

## Table of Contents  
- [Introduction](#introduction)  
- [Technologies and Libraries](#technologies-and-libraries)  
- [AI Techniques and Models Used](#ai-techniques-and-models-used)  
- [Datasets](#datasets)  
- [Data Preprocessing](#data-preprocessing)  
- [Embedding Generation](#embedding-generation)  
- [Similarity Calculation and Matching](#similarity-calculation-and-matching)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Visualization](#visualization)  
- [Streamlit Web Application](#streamlit-web-application)  
- [How to Run](#how-to-run)  
- [Project Structure](#project-structure)  
- [Conclusion](#conclusion)  

---

## Introduction  
The system addresses the challenge of matching resumes with relevant job descriptions beyond simple keyword searches. Using natural language understanding through sentence embeddings, it produces semantically aware recommendations, improving HR recruitment workflows and enabling better candidate-job fit.

---

## Technologies and Libraries  

- **Programming Language:** Python 3.x  
- **Data Handling:** Pandas, NumPy  
- **Natural Language Processing:**  
  - `sentence-transformers` for pretrained transformer embedding models  
  - Regex for text cleaning and normalization  
- **Similarity Computation:** Scikit-learn (cosine similarity)  
- **Visualization:** Matplotlib, Seaborn, WordCloud  
- **PDF Parsing:** PyMuPDF (fitz) for extracting text from PDF resumes (optional)  
- **Web Framework:** Streamlit for interactive UI and filtering  
- **Model Saving:** Joblib for model and embeddings serialization  

---

## AI Techniques and Models Used  

- **Transformer-based Sentence Embeddings:**  
  - Primary model: `sentence-transformers/all-MiniLM-L6-v2`  
  - This model converts job descriptions and resumes into dense 384-dimensional vectors capturing semantic meaning.  
- **Cosine Similarity:**  
  - Used to measure closeness between job and resume embeddings for ranking matches.  
- **Evaluation Metrics:**  
  - Top-K Accuracy, Precision@K, Recall@K, Mean Reciprocal Rank (MRR) for recommendation quality  
  - Binary classification metrics (Accuracy, Precision, Recall, F1, ROC AUC) on thresholded similarity scores  
- **Threshold Tuning:**  
  - Optimizing similarity threshold for best balance of precision and recall.  

---

## Datasets  

- **Job Postings Dataset (`postings_sample.csv`):**  
  Contains job metadata like job titles, descriptions, companies, domains, locations, and skills.  
- **Resumes Dataset (`resume_sample.csv`):**  
  Includes candidate resumes with raw text, processed text, categorized skills, and domains.  

---

## Data Preprocessing  

- Filled missing values in key textual fields to avoid errors during embedding.  
- Normalized text data: converted to lowercase, removed special characters and punctuation using regex.  
- Combined fields where needed (e.g., job title + description) for comprehensive text representation.  
- Extracted and retained relevant metadata such as job domain and candidate skills for filtering.  

---

## Embedding Generation  

- Loaded pre-trained `all-MiniLM-L6-v2` model from Sentence Transformers.  
- Generated fixed-length semantic embeddings for each job description and resume text.  
- Stored embeddings in NumPy arrays for efficient similarity computations.  

---

## Similarity Calculation and Matching  

- Computed cosine similarity matrix between all job and resume embeddings.  
- Ranked resumes per job by similarity score to recommend top candidates.  
- Enabled filtering in the Streamlit app by location, domain, and experience to refine search.  

---

## Evaluation Metrics  

- Implemented Top-K Accuracy, Precision@K, Recall@K, and MRR to evaluate recommendation performance.  
- Converted similarity scores to binary matches using a tuned threshold and calculated Accuracy, Precision, Recall, F1 Score, and ROC AUC.  
- Used these metrics to understand model effectiveness in real-world matching scenarios.  

---

## Visualization  

- Bar plots for distribution of job postings by domain, location, and company.  
- Pie charts for categorical breakdowns of job types and candidate domains.  
- Word clouds highlighting most frequent keywords in job titles and resume texts.  
- Histograms of similarity score distributions for insight into matching confidence.  

---

## Streamlit Web Application  

- Developed an interactive UI for users to:  
  - Upload resume PDFs (optional) and extract text using PyMuPDF.  
  - Select filters like job domain, location, and experience level to tailor recommendations.  
  - View top recommended jobs with similarity scores and company logos.  
  - Expand job descriptions and provide feedback via upvote/downvote buttons.  
- The app enhances user experience by providing real-time, personalized job recommendations.  

---

## How to Run  

### Prerequisites  
- Python 3.7 or higher  
- Recommended: Create and activate a virtual environment  

### Install Dependencies  
```bash
pip install -r requirements.txt
````

### Run Streamlit App

```bash
streamlit run app.py
```

### Run Embedding & Evaluation Scripts

* Execute `job_recommender.py` or your main script to:

  * Preprocess data
  * Generate embeddings
  * Compute similarity
  * Visualize results
  * Evaluate performance

---



## Requirements.txt Example

```
pandas
numpy
sentence-transformers
scikit-learn
matplotlib
seaborn
wordcloud
joblib
PyMuPDF
streamlit
regex
```

---

## Conclusion

This project demonstrates a practical AI system for automating and improving the candidate-job matching process using state-of-the-art sentence embedding models. The semantic approach captures deeper contextual meaning, overcoming traditional keyword limitations and enabling HR teams to efficiently surface qualified candidates. The combination of thorough preprocessing, scalable embedding computation, intuitive visualizations, and interactive web UI provides a robust foundation for real-world recruitment enhancement.

---


```

