
import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# --- Page config ---
st.set_page_config(page_title="Smart Job Recommender", layout="wide")
st.markdown("""
<style>
.reportview-container { background-color: #f9f9f9; }
.sidebar .sidebar-content { background: #f0f2f6; }
.stButton>button { background-color: #4CAF50; color: white; }
.score { font-size: 14px; font-weight: bold; color: #555; }
</style>
""", unsafe_allow_html=True)

# --- Load dataset and model ---
@st.cache_data
def load_jobs():
    return pd.read_csv("postings_sample.csv")

job_df = load_jobs()
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Sidebar UI ---
st.sidebar.title("📄 Upload Your Resume")
uploaded_resume = st.sidebar.file_uploader("Upload Resume PDF", type=["pdf"])

# --- Optional Filters ---
st.sidebar.markdown("## 🧩 Filters")

# ✅ Added filters below
work_type = st.sidebar.selectbox("🏢 Work Type", options=["All"] + sorted(job_df['work_type'].dropna().unique()) if 'work_type' in job_df else ["All"])
job_domain = st.sidebar.selectbox("🧠 Job Domain", options=["All"] + sorted(job_df['job_domain'].dropna().unique()) if 'job_domain' in job_df else ["All"])
location = st.sidebar.selectbox("🌍 Location", options=["All"] + sorted(job_df['location'].dropna().unique()) if 'location' in job_df else ["All"])
title = st.sidebar.selectbox("📌 Job Title", options=["All"] + sorted(job_df['title'].dropna().unique()) if 'title' in job_df else ["All"])
company = st.sidebar.selectbox("🏢 Company", options=["All"] + sorted(job_df['company_name'].dropna().unique()) if 'company_name' in job_df else ["All"])

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def preprocess_text(text):
    import re
    return {'lemmatized': re.sub(r'\W+', ' ', text.lower()).strip()}

def get_logo_base64(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# --- Main Area ---
st.title("💼 Smart Job Recommendation System")
st.markdown("Get job recommendations by uploading your resume.")

# --- If Resume Uploaded ---
if uploaded_resume is not None:
    with st.spinner("🔍 Extracting and analyzing resume..."):
        raw_text = extract_text_from_pdf(uploaded_resume)
        resume_text = preprocess_text(raw_text)
        resume_embedding = model.encode([resume_text['lemmatized']])

        # Preview resume
        with st.expander("📄 Preview Resume Text"):
            st.code(raw_text[:1500] + "..." if len(raw_text) > 1500 else raw_text, language='markdown')

        # Apply filters
        filtered_df = job_df.copy()
        if work_type != "All":
            filtered_df = filtered_df[filtered_df["work_type"] == work_type]
        if job_domain != "All":
            filtered_df = filtered_df[filtered_df["job_domain"] == job_domain]
        if location != "All":
            filtered_df = filtered_df[filtered_df["location"] == location]
        if title != "All":
            filtered_df = filtered_df[filtered_df["title"] == title]
        if company != "All":
            filtered_df = filtered_df[filtered_df["company_name"] == company]

        if filtered_df.empty:
            st.warning("No jobs match your selected filters.")
        else:
            # Get top 5 matches
            job_titles = filtered_df['title'].astype(str).tolist()
            job_embeddings = model.encode(job_titles)
            similarities = cosine_similarity(resume_embedding, job_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:5]

            st.success("🎯 Found top matching jobs!")
            st.markdown("### 🔝 Recommended Jobs")

            for idx in top_indices:
                job = filtered_df.iloc[idx]
                title = job['title']
                similarity = similarities[idx]
                desc = job.get('description', 'No description available.')
                company_name = job.get('company_name', 'Unknown Company')

                # Logo (if available)
                logo_html = ""
                if 'Logo' in job and pd.notna(job['Logo']):
                    base64_img = get_logo_base64(job['Logo'])
                    if base64_img:
                        logo_html = f'<img src="data:image/png;base64,{base64_img}" style="height:40px;margin-right:10px;">'

                st.markdown(f"""
<div style="display:flex;align-items:center;">
    {logo_html}<h4 style="margin:0;">📌 {title}</h4>
</div>
<p class="score">🏢 {company_name} </p>
""", unsafe_allow_html=True)

                with st.expander("🔽 View Job Description"):
                    st.write(desc.strip())

                # Feedback buttons
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"👍 Relevant Match {idx}", key=f"up{idx}"):
                        st.success("Thanks for your feedback!")
                with col2:
                    if st.button(f"👎 Not Relevant {idx}", key=f"down{idx}"):
                        st.info("Got it! We'll improve future matches.")

                st.markdown("---")

else:
    st.info("📎 Please upload a resume PDF to begin.") 
