"""
AI Brain for Modern Hiring — Sandbox Demo
==========================================
Team Insomniaks | Redrob Hackathon

Full end-to-end ranking pipeline:
  FAISS Semantic Retrieval → ConsultantBERT Classification → Composite Scoring
"""

import streamlit as st
import json
import os
import io
import csv
import time

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Brain for Modern Hiring | Team Insomniaks",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp { font-family: 'Inter', sans-serif; }

/* ---- Hero ---- */
.hero-title {
    font-size: 2.8rem; font-weight: 800; line-height: 1.15;
    background: linear-gradient(135deg, #00C9A7 0%, #845EC2 50%, #D65DB1 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-sub { color: #8B949E; font-size: 1.05rem; margin-bottom: 1.5rem; }

/* ---- Stat cards ---- */
.stat-card {
    background: linear-gradient(135deg, rgba(0,201,167,.08), rgba(132,94,194,.06));
    border: 1px solid rgba(0,201,167,.18); border-radius: 14px;
    padding: 1.2rem 1.4rem; text-align: center;
}
.stat-card .num { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: #00C9A7; }
.stat-card .lbl { font-size: .82rem; color: #8B949E; margin-top: .25rem; }

/* ---- Pipeline step ---- */
.pipe-step {
    background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 1rem 1.3rem; margin: .4rem 0;
    display: flex; align-items: center; gap: .8rem;
}
.pipe-step .icon { font-size: 1.5rem; }
.pipe-step .title { font-weight: 600; font-size: .95rem; }
.pipe-step .desc { font-size: .8rem; color: #8B949E; }

/* ---- Candidate card ---- */
.cand-card {
    background: rgba(255,255,255,.025); border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px; padding: 1.3rem 1.5rem; margin: .6rem 0;
    transition: border-color .25s, transform .25s;
}
.cand-card:hover { border-color: rgba(0,201,167,.35); transform: translateY(-2px); }
.cand-rank {
    display: inline-flex; align-items: center; justify-content: center;
    width: 2.2rem; height: 2.2rem; border-radius: 50%;
    background: linear-gradient(135deg, #00C9A7, #845EC2);
    color: #fff; font-weight: 700; font-size: .9rem; margin-right: .8rem;
}
.cand-name { font-weight: 700; font-size: 1.1rem; }
.cand-role { color: #8B949E; font-size: .85rem; }

/* ---- Skill tags ---- */
.skill-tag {
    display: inline-block; padding: .18rem .55rem;
    background: rgba(132,94,194,.14); color: #B39DDB;
    border-radius: 6px; font-size: .72rem; margin: .12rem; font-weight: 500;
}

/* ---- Score pills ---- */
.score-pill {
    display: inline-block; padding: .2rem .65rem;
    border-radius: 20px; font-weight: 600; font-size: .82rem;
    font-family: 'JetBrains Mono', monospace;
}
.score-high { background: rgba(0,201,167,.14); color: #00C9A7; }
.score-mid  { background: rgba(255,193,7,.14);  color: #FFD54F; }
.score-low  { background: rgba(255,82,82,.14);  color: #FF5252; }

/* ---- Hide defaults ---- */
#MainMenu {visibility: hidden;} footer {visibility: hidden;}

/* ---- Metric mono ---- */
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

/* ---- Divider ---- */
.section-divider {
    border: none; border-top: 1px solid rgba(255,255,255,.06);
    margin: 2rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "final_consultant_bert").replace("\\", "/")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_candidates.jsonl")

# ── Model Loading (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load SentenceTransformer + ConsultantBERT. Cached across sessions."""
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not os.path.isdir(MODEL_DIR):
        return None, None, None, (
            "**ConsultantBERT model not found.**\n\n"
            "Please download the `final_consultant_bert/` folder from the "
            "[Google Drive](https://drive.google.com/drive/folders/"
            "1N2ImPewxUVoKemlJTyqLlRg9r8WrTMQ6?usp=drive_link) "
            "and place it in the app directory."
        )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    bert = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    bert.eval()
    return embedder, tok, bert, None


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_sample_candidates():
    """Load the pre-bundled 100-candidate sample."""
    if not os.path.exists(SAMPLE_PATH):
        return None
    cands = []
    with open(SAMPLE_PATH, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                cands.append(json.loads(line))
    return cands


def parse_uploaded_jsonl(uploaded_file):
    """Parse an uploaded JSONL file into a list of candidate dicts."""
    text = uploaded_file.getvalue().decode("utf-8")
    cands = []
    for line in text.splitlines():
        if line.strip():
            cands.append(json.loads(line))
    return cands


# ── Pipeline Functions ───────────────────────────────────────────────────────

def _build_faiss_index(candidates, embedder):
    """Build a small in-memory FAISS index from candidate profiles."""
    import faiss
    import numpy as np

    texts, mapping = [], []
    for cand in candidates:
        p = cand.get("profile", {})
        sk = cand.get("skills", [])
        skill_str = " ".join(
            s.get("name", "") if isinstance(s, dict) else str(s) for s in sk
        )
        texts.append(f"{p.get('headline', '')} {p.get('summary', '')} {skill_str}")
        mapping.append(cand["candidate_id"])

    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embs)

    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx, mapping


def run_pipeline(jd_text, candidates, embedder, tokenizer, bert_model, progress):
    """
    Execute the full ranking pipeline (mirrors rank.py):
      1. Build FAISS index from candidates
      2. Encode JD → FAISS semantic search
      3. Honeypot / adversarial filtering
      4. ConsultantBERT classification
      5. Composite scoring & ranking
    Returns dict with 'rankings', 'stages', 'stats'.
    """
    import faiss
    import torch

    stages, stats = {}, {}

    # ── Stage 1: Build FAISS index ──
    progress.progress(0.10, text="🔧  Stage 1/5 — Building FAISS index from candidate embeddings …")
    time.sleep(0.3)  # small pause for UI smoothness
    idx, mapping = _build_faiss_index(candidates, embedder)
    stats["total_candidates"] = len(candidates)
    stages["1_index"] = f"Indexed {len(candidates)} candidate embeddings ({idx.d}-dim)"

    # ── Stage 2: Encode JD & search ──
    progress.progress(0.30, text="🔍  Stage 2/5 — Encoding job description & searching FAISS …")
    jd_emb = embedder.encode([jd_text], convert_to_numpy=True)
    faiss.normalize_L2(jd_emb)

    k = min(500, len(candidates))
    dists, idxs = idx.search(jd_emb, k)

    sem_scores = {}
    for d, i in zip(dists[0], idxs[0]):
        if 0 <= int(i) < len(mapping):
            sem_scores[mapping[int(i)]] = float(d)

    stats["faiss_retrieved"] = len(sem_scores)
    stages["2_search"] = f"Retrieved top-{len(sem_scores)} semantic matches"

    lookup = {c["candidate_id"]: c for c in candidates}
    shortlist = [lookup[cid] for cid in sem_scores if cid in lookup]

    # ── Stage 3: ConsultantBERT classification ──
    progress.progress(0.65, text="🤖  Stage 3/4 — Running ConsultantBERT on each candidate …")
    rankings = []
    total = len(shortlist)
    for ci, cand in enumerate(shortlist):
        # Update sub-progress every 10 candidates
        if ci % 10 == 0:
            frac = 0.65 + 0.25 * (ci / max(total, 1))
            progress.progress(
                min(frac, 0.90),
                text=f"🤖  Stage 3/4 — ConsultantBERT inference … ({ci}/{total})",
            )

        cid = cand["candidate_id"]
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})

        txt = f"{profile.get('headline', '')} {profile.get('summary', '')}"
        inputs = tokenizer(txt[:512], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = bert_model(**inputs)
            probs = torch.nn.functional.softmax(out.logits, dim=-1)
            product_score = probs[0][1].item()

        resp_rate = signals.get("recruiter_response_rate", 0.5)
        sem = sem_scores.get(cid, 0.5)
        exp_yrs = profile.get("years_of_experience", 0)
        exp_score = min(exp_yrs / 7.0, 1.0)

        # Composite (matching rank.py weights: 0.40 semantic + 0.30 BERT + 0.20 experience + 0.10 behavioral)
        composite = (sem * 0.40) + (product_score * 0.30) + (exp_score * 0.20) + (resp_rate * 0.10)

        reasoning = (
            f"Matched with {exp_yrs} yrs exp. "
            f"Product-fit prob {product_score:.2f}. "
            f"Response rate {resp_rate:.2f}."
        )

        name = (
            cand.get("name")
            or profile.get("anonymized_name")
            or profile.get("name")
            or f"Candidate {cid[-4:]}"
        )
        role = profile.get("headline", profile.get("current_title", "Engineer"))
        summary = profile.get("summary", "")[:200]
        skill_names = [
            s.get("name", "") if isinstance(s, dict) else str(s)
            for s in cand.get("skills", [])
        ][:6]

        rankings.append(
            {
                "candidate_id": cid,
                "rank": 0,
                "name": name,
                "role": role,
                "summary": summary,
                "skills": skill_names,
                "years_of_experience": exp_yrs,
                "score": round(composite, 4),
                "semantic_score": round(sem, 4),
                "product_fit_score": round(product_score, 4),
                "response_rate": round(resp_rate, 4),
                "reasoning": reasoning,
            }
        )

    stages["3_bert"] = f"Classified {len(rankings)} candidates via ConsultantBERT"

    # ── Stage 4: Sort & rank ──
    progress.progress(0.95, text="📊  Stage 4/4 — Computing composite scores & final ranking …")
    rankings.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    for i, r in enumerate(rankings, 1):
        r["rank"] = i

    stats["final_ranked"] = len(rankings)
    stages["4_rank"] = f"Final ranking: {len(rankings)} candidates sorted by composite score"

    progress.progress(1.0, text="✅  Pipeline complete!")
    return {"rankings": rankings, "stages": stages, "stats": stats}


def rankings_to_csv_bytes(rankings):
    """Serialize rankings to CSV bytes for download."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rankings:
        w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
    return buf.getvalue().encode("utf-8")


def _score_class(val):
    if val >= 0.7:
        return "score-high"
    if val >= 0.4:
        return "score-mid"
    return "score-low"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 AI Brain")
    st.caption("Team Insomniaks · Redrob Hackathon")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🧪 Ranking Sandbox", "📊 Dataset Explorer", "🏗️ Architecture"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "##### 📦 Required Assets\n"
        "`final_consultant_bert/` model must be present for live inference.\n\n"
        "[📥 Download from Google Drive]"
        "(https://drive.google.com/drive/folders/1N2ImPewxUVoKemlJTyqLlRg9r8WrTMQ6)"
    )
    st.markdown("---")
    st.caption("Built with ❤️ by Team Insomniaks")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown('<h1 class="hero-title">AI Brain for Modern Hiring</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">'
        "Hybrid Multi-Stage Candidate Ranking Funnel — processing 100K+ profiles "
        "through semantic retrieval, adversarial filtration, domain classification, "
        "and composite scoring."
        "</p>",
        unsafe_allow_html=True,
    )

    # Status indicator
    with st.spinner("Loading ML models …"):
        embedder, tokenizer, bert_model, err = load_models()

    if err:
        st.error(err)
    else:
        st.success("✅ All ML models loaded and ready for inference.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="stat-card"><div class="num">100K+</div>'
            '<div class="lbl">Candidate Profiles</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="stat-card"><div class="num">384-d</div>'
            '<div class="lbl">Embedding Vectors</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="stat-card"><div class="num">4</div>'
            '<div class="lbl">Pipeline Stages</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="stat-card"><div class="num">&lt;5 min</div>'
            '<div class="lbl">CPU Inference</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### 🔬 Pipeline Stages")
    steps = [
        ("🔍", "FAISS Semantic Retrieval", "Encode JD with MiniLM-L6-v2, search pre-indexed candidate embeddings"),
        ("🤖", "ConsultantBERT Classification", "Fine-tuned DistilBERT distinguishes product engineers from consultants"),
        ("📊", "Composite Signal Weighting", "0.40×semantic + 0.30×BERT + 0.20×experience + 0.10×behavioral"),
        ("🏆", "Final Ranking & CSV Export", "Sort by composite score, break ties by candidate ID, export top-N"),
    ]
    for icon, title, desc in steps:
        st.markdown(
            f'<div class="pipe-step">'
            f'<span class="icon">{icon}</span>'
            f'<div><div class="title">{title}</div><div class="desc">{desc}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### 👥 Team Insomniaks")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Shaurya Aggarwal** — ML Pipeline & Backend")
    with t2:
        st.markdown("**Navyaa Mittal** — Frontend & Design")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RANKING SANDBOX
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧪 Ranking Sandbox":
    st.markdown('<h1 class="hero-title">Ranking Sandbox</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">'
        "Paste any Job Description, run the full pipeline end-to-end, "
        "and download the ranked CSV."
        "</p>",
        unsafe_allow_html=True,
    )

    # Load models
    with st.spinner("Loading ML models …"):
        embedder, tokenizer, bert_model, model_err = load_models()

    if model_err:
        st.error(model_err)
        st.stop()

    # ── Input section ──
    st.markdown("### 📝 Job Description")
    jd_text = st.text_area(
        "Enter the Job Description to rank candidates against:",
        value=(
            "Senior AI Engineer. Production experience with embeddings-based retrieval systems, "
            "vector databases (Pinecone, FAISS), and strong Python. "
            "Needs evaluation framework experience (NDCG, MRR). "
            "Must have product company experience, not pure consulting or pure academic research."
        ),
        height=140,
        label_visibility="collapsed",
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Candidate input ──
    st.markdown("### 📂 Candidate Data")
    data_mode = st.radio(
        "Choose input mode:",
        ["Use pre-loaded sample (100 candidates)", "Upload your own JSONL file"],
        horizontal=True,
    )

    candidates = None
    if data_mode.startswith("Use pre-loaded"):
        candidates = load_sample_candidates()
        if candidates:
            st.info(f"✅ Loaded **{len(candidates)}** pre-bundled sample candidates.")
        else:
            st.error(
                "`sample_candidates.jsonl` not found. "
                "Please upload a JSONL file or add the sample file to the app directory."
            )
    else:
        uploaded = st.file_uploader(
            "Upload candidates.jsonl (≤100 candidates recommended)",
            type=["jsonl", "json"],
        )
        if uploaded:
            try:
                candidates = parse_uploaded_jsonl(uploaded)
                st.info(f"✅ Parsed **{len(candidates)}** candidates from upload.")
            except Exception as exc:
                st.error(f"Failed to parse file: {exc}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Run button ──
    col_run, col_space = st.columns([1, 3])
    with col_run:
        run_btn = st.button(
            "🚀 Run Ranking Pipeline",
            type="primary",
            use_container_width=True,
            disabled=(candidates is None),
        )

    if run_btn and candidates:
        st.markdown("### ⚙️ Pipeline Execution")
        progress = st.progress(0, text="Initializing …")

        t0 = time.time()
        result = run_pipeline(jd_text, candidates, embedder, tokenizer, bert_model, progress)
        elapsed = time.time() - t0

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Stats row ──
        st.markdown("### 📈 Execution Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Input", result["stats"]["total_candidates"])
        s2.metric("FAISS Hits", result["stats"]["faiss_retrieved"])
        s3.metric("Final Ranked", result["stats"]["final_ranked"])
        s4.metric("Time", f"{elapsed:.1f}s")

        # Stage log
        with st.expander("📋 Pipeline Stage Log", expanded=False):
            for key in sorted(result["stages"]):
                st.markdown(f"- **{key}**: {result['stages'][key]}")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── CSV Download ──
        csv_bytes = rankings_to_csv_bytes(result["rankings"])
        st.download_button(
            label="📥 Download Ranked CSV (team_insomniaks.csv)",
            data=csv_bytes,
            file_name="team_insomniaks.csv",
            mime="text/csv",
            type="primary",
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Results display ──
        st.markdown(f"### 🏆 Top Ranked Candidates ({len(result['rankings'])})")

        for r in result["rankings"]:
            sc = _score_class(r["score"])
            skills_html = " ".join(
                f'<span class="skill-tag">{s}</span>' for s in r["skills"] if s
            )

            st.markdown(
                f"""
<div class="cand-card">
  <div style="display:flex; align-items:center; margin-bottom:.6rem;">
    <span class="cand-rank">{r['rank']}</span>
    <div>
      <span class="cand-name">{r['name']}</span><br>
      <span class="cand-role">{r['role']}</span>
    </div>
    <div style="margin-left:auto;">
      <span class="score-pill {sc}">{r['score']:.4f}</span>
    </div>
  </div>
  <div style="margin-bottom:.5rem; font-size:.85rem; color:#8B949E;">{r['summary']}</div>
  <div style="margin-bottom:.5rem;">{skills_html}</div>
  <div style="display:flex; gap:1.5rem; font-size:.78rem; color:#8B949E;">
    <span>🔍 Semantic: <b style="color:#E6EDF3">{r['semantic_score']:.3f}</b></span>
    <span>🤖 BERT Fit: <b style="color:#E6EDF3">{r['product_fit_score']:.3f}</b></span>
    <span>📊 Response: <b style="color:#E6EDF3">{r['response_rate']:.3f}</b></span>
    <span>📅 Exp: <b style="color:#E6EDF3">{r['years_of_experience']} yrs</b></span>
  </div>
  <div style="margin-top:.4rem; font-size:.78rem; color:#6E7681; font-style:italic;">
    {r['reasoning']}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Dataset Explorer":
    st.markdown('<h1 class="hero-title">Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Browse the pre-loaded candidate sample.</p>',
        unsafe_allow_html=True,
    )

    candidates = load_sample_candidates()
    if not candidates:
        st.error("`sample_candidates.jsonl` not found.")
        st.stop()

    # Search / filter
    search = st.text_input("🔎 Search by skill, role, or keyword …", "")
    col_exp, _ = st.columns([1, 3])
    with col_exp:
        min_exp = st.slider("Min years of experience", 0, 30, 0)

    filtered = candidates
    if search:
        sl = search.lower()
        filtered = [
            c
            for c in filtered
            if sl in json.dumps(c).lower()
        ]
    if min_exp > 0:
        filtered = [
            c for c in filtered
            if c.get("profile", {}).get("years_of_experience", 0) >= min_exp
        ]

    st.caption(f"Showing **{len(filtered)}** of {len(candidates)} candidates")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    for cand in filtered:
        p = cand.get("profile", {})
        name = p.get("anonymized_name", cand.get("name", cand["candidate_id"]))
        role = p.get("headline", p.get("current_title", "—"))
        summary = p.get("summary", "")[:180]
        exp = p.get("years_of_experience", "?")
        skills = cand.get("skills", [])
        skill_html = " ".join(
            f'<span class="skill-tag">{s.get("name","") if isinstance(s,dict) else s}</span>'
            for s in skills[:8]
        )
        signals = cand.get("redrob_signals", {})
        resp = signals.get("recruiter_response_rate", "—")

        st.markdown(
            f"""
<div class="cand-card">
  <div style="display:flex; justify-content:space-between; align-items:start;">
    <div>
      <span class="cand-name">{name}</span><br>
      <span class="cand-role">{role}</span>
    </div>
    <div style="text-align:right; font-size:.82rem; color:#8B949E;">
      <div>📅 {exp} yrs experience</div>
      <div>📊 Response rate: {resp if isinstance(resp, str) else f"{resp:.0%}"}</div>
    </div>
  </div>
  <div style="margin:.5rem 0; font-size:.85rem; color:#8B949E;">{summary}</div>
  <div>{skill_html}</div>
</div>
""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏗️ Architecture":
    st.markdown('<h1 class="hero-title">System Architecture</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">'
        "Deep dive into the multi-stage ranking funnel."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🔄 Pipeline Flow")
    st.markdown(
        """
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Job Description (input text)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: FAISS Semantic Retrieval                                  │
│  ├─ Encode JD with all-MiniLM-L6-v2 (384-dim)                     │
│  ├─ L2-normalize embeddings                                        │
│  └─ Inner-product search → Top 500 candidates                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: ConsultantBERT Classification                             │
│  ├─ Custom fine-tuned DistilBERT (2 classes)                       │
│  ├─ Input: headline + summary (truncated to 512 tokens)            │
│  └─ Output: P(Product Engineer) probability                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Composite Score                                           │
│  score = 0.40 × semantic + 0.30 × BERT + 0.20 × exp + 0.10 × beh  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4: Sort & Export                                             │
│  ├─ Sort descending by composite score                             │
│  ├─ Tie-break by candidate_id ascending                            │
│  └─ Export top-100 → team_insomniaks.csv                           │
└─────────────────────────────────────────────────────────────────────┘
```
"""
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### 🧠 ConsultantBERT")
    st.markdown(
        """
Our custom **ConsultantBERT** model is a fine-tuned **DistilBERT** binary classifier trained to
distinguish between:

| Label | Description |
|-------|-------------|
| **0 — IT Services / Consulting** | Candidates with primarily outsourcing, body-shopping, or consulting backgrounds |
| **1 — Product Engineering** | Candidates with genuine product-company engineering experience |

**Training details:**
- Base model: `distilbert-base-uncased`
- Training data: Labeled subset from the Redrob candidate pool
- Features: Concatenated `headline + summary` text
- Training: Standard cross-entropy loss, AdamW optimizer

The model is stored locally in `final_consultant_bert/` and loaded with `local_files_only=True`
to ensure zero network dependency during inference.
"""
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### ⚖️ Scoring Weights")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown(
            '<div class="stat-card"><div class="num">0.40</div>'
            '<div class="lbl">Semantic Similarity</div></div>',
            unsafe_allow_html=True,
        )
    with w2:
        st.markdown(
            '<div class="stat-card"><div class="num">0.30</div>'
            '<div class="lbl">BERT Product-Fit</div></div>',
            unsafe_allow_html=True,
        )
    with w3:
        st.markdown(
            '<div class="stat-card"><div class="num">0.20</div>'
            '<div class="lbl">Experience Signal</div></div>',
            unsafe_allow_html=True,
        )
    with w4:
        st.markdown(
            '<div class="stat-card"><div class="num">0.10</div>'
            '<div class="lbl">Behavioral Signal</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### 📓 Offline Training")
    st.markdown(
        "The complete offline pipeline — data labeling, ConsultantBERT fine-tuning, "
        "embedding generation, and FAISS index construction — is documented in:\n\n"
        "`offline_training/offline_pipeline.ipynb`"
    )
