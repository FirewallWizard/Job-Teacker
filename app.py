import streamlit as st
import pandas as pd
import uuid
import os
import requests
import re
from bs4 import BeautifulSoup
from datetime import timedelta
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# OPENAI (MODERN CLIENT)
# =========================
from openai import OpenAI

def get_openai_client():
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY", None)
        or st.session_state.get("OPENAI_API_KEY", None)
    )
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# =========================
# CONFIG
# =========================
EXCEL_FILE = "job_applications.xlsx"
RESUME_FILE = "resume.txt"

st.set_page_config("Job Tracker CRM", layout="wide")

# =========================
# MODEL (LOAD ONCE)
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# UTILITIES
# =========================
def ensure_columns(df, cols):
    for c, d in cols.items():
        if c not in df.columns:
            df[c] = d
    return df


def load_resume():
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def save_resume(text):
    with open(RESUME_FILE, "w", encoding="utf-8") as f:
        f.write(text)


def load_jobs():
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE, "Jobs")
        except:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    return ensure_columns(df, {
        "Job_ID": "",
        "Company": "",
        "Position": "",
        "Status": "Applied",
        "Source": "",
        "Job_Link": "",
        "Job_Description": "",
        "Is_Archived": False,
    })


def load_contacts():
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE, "Contacts")
        except:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    return ensure_columns(df, {
        "Contact_ID": "",
        "Job_ID": "",
        "Company": "",
        "Name": "",
        "Role": "",
        "Outreach_Type": "",
        "Follow_Up_Date": "",
        "Response_Status": "No Response",
        "Notes": "",
    })


def save_all(jobs, contacts):
    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl", mode="w") as w:
        jobs.to_excel(w, "Jobs", index=False)
        contacts.to_excel(w, "Contacts", index=False)


def extract_job_details(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.title.text if soup.title else ""
    desc = " ".join(p.get_text() for p in soup.find_all(["p", "li"]))[:5000]
    return title, desc


# =========================
# MATCHING + ANALYSIS
# =========================
def match_score(resume, jd):
    if not resume or not jd:
        return None
    emb = model.encode([resume, jd], normalize_embeddings=True)
    return round(cosine_similarity([emb[0]], [emb[1]])[0][0] * 100, 2)


def score_color(score):
    if score is None:
        return "—"
    if score >= 80:
        return f"🟢 {score}%"
    if score >= 60:
        return f"🟡 {score}%"
    return f"🔴 {score}%"


def extract_keywords(text, n=15):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop = {"the","and","for","with","that","this","from","you","your","are"}
    return [w for w, _ in Counter(w for w in words if w not in stop).most_common(n)]


def low_match_reason(resume, jd):
    missing = set(extract_keywords(jd)) - set(extract_keywords(resume))
    return ", ".join(list(missing)[:8]) if missing else "Good alignment"


# =========================
# OPENAI PROMPT
# =========================
RESUME_TAILOR_PROMPT = """
You are an expert resume writer.

Given a resume and job description:
- Identify missing skills
- Suggest resume improvements
- Rewrite bullet points where useful
- Optimize for ATS
- Be truthful (no fabrication)

Output:
- Missing Skills
- Resume Improvements
- Optional Summary Rewrite

Resume:
{resume}

Job Description:
{job_description}
"""

@st.cache_data(show_spinner=False)
def generate_resume_suggestions(resume, jd):
    client = get_openai_client()
    if not client:
        return "❌ OpenAI API key not configured."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": RESUME_TAILOR_PROMPT.format(
                resume=resume, job_description=jd
            )}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content


# =========================
# LOAD DATA
# =========================
jobs_df = load_jobs()
contacts_df = load_contacts()
resume_text = load_resume()
today = pd.Timestamp.today().normalize()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""

if not get_openai_client():
    st.sidebar.warning("OpenAI API Key missing")
    st.session_state.OPENAI_API_KEY = st.sidebar.text_input(
        "Enter OpenAI API Key", type="password"
    )

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Resume", "Add Job", "Applications", "Contacts", "Resume Tailor 🧠"]
)

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    st.title("📊 Follow-Up Dashboard")
    contacts_df["Follow_Up_Date"] = pd.to_datetime(
        contacts_df["Follow_Up_Date"], errors="coerce"
    )
    st.metric("Overdue",
        len(contacts_df[contacts_df["Follow_Up_Date"] < today])
    )

# =========================
# RESUME PAGE
# =========================
elif page == "Resume":
    st.title("📄 Resume Intelligence")
    resume_text = st.text_area("Resume", resume_text, height=350)
    if st.button("Save Resume"):
        save_resume(resume_text)
        st.success("Saved")

# =========================
# ADD JOB
# =========================
elif page == "Add Job":
    st.title("➕ Add Job")
    company = st.text_input("Company")
    position = st.text_input("Position")
    jd = st.text_area("Job Description")
    if st.button("Add"):
        jobs_df = pd.concat([jobs_df, pd.DataFrame([{
            "Job_ID": str(uuid.uuid4()),
            "Company": company,
            "Position": position,
            "Job_Description": jd
        }])])
        save_all(jobs_df, contacts_df)
        st.rerun()

# =========================
# APPLICATIONS
# =========================
elif page == "Applications":
    rows = []
    for _, j in jobs_df.iterrows():
        s = match_score(resume_text, j["Job_Description"])
        rows.append({
            "Company": j["Company"],
            "Position": j["Position"],
            "Match": score_color(s),
            "Why Low Match": low_match_reason(resume_text, j["Job_Description"])
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =========================
# CONTACTS
# =========================
elif page == "Contacts":
    rows = []
    for _, c in contacts_df.iterrows():
        jd = jobs_df[jobs_df["Job_ID"] == c["Job_ID"]]["Job_Description"].values
        s = match_score(resume_text, jd[0] if len(jd) else "")
        rows.append({**c, "Resume Match": score_color(s)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =========================
# RESUME TAILOR (AI)
# =========================
elif page == "Resume Tailor 🧠":
    st.title("🧠 Resume Improvement Suggestions")

    job_map = {
        f"{j.Company} — {j.Position}": j.Job_ID
        for _, j in jobs_df.iterrows()
    }
    selected = st.selectbox("Select Job", job_map.keys())
    job = jobs_df[jobs_df["Job_ID"] == job_map[selected]].iloc[0]

    if st.button("Generate Suggestions"):
        with st.spinner("Analyzing resume..."):
            output = generate_resume_suggestions(
                resume_text,
                job["Job_Description"]
            )
        st.markdown(output)
