---
title: AI Brain for Modern Hiring
emoji: 🧠
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: "1.45.0"
app_file: streamlit_app.py
pinned: false
license: mit
---

# Redrob Candidate Ranking - Team Insomniaks

## Overview

This repository contains Team **Insomniaks'** submission for the **Redrob Hackathon**. Our solution implements a **Hybrid Multi-Stage Candidate Ranking Funnel** designed to efficiently process over **100,000 candidate profiles** and return the **Top 100 most relevant candidates** in **under 5 minutes** on a standard CPU-only machine with limited memory.

The system combines semantic retrieval, deterministic validation, domain-specific classification, and composite ranking to achieve high accuracy while remaining computationally efficient and robust against adversarial or misleading candidate profiles.

---

## Architecture Overview

The `rank.py` script executes a highly optimized and deterministic multi-stage ranking pipeline:

### 1. Semantic Retrieval using FAISS

The input Job Description (JD) is embedded at runtime using the `all-MiniLM-L6-v2` sentence transformer model. The embedding is then used to query a pre-computed FAISS index containing embeddings for all candidate profiles.

Instead of evaluating all 100,000 candidates, the system instantly retrieves the **Top 500 semantically similar candidates**, significantly reducing both runtime and memory usage while preserving retrieval quality.

---

### 2. Honeypot and Inconsistency Filtration

The shortlisted candidates are passed through a deterministic rule-based filtration layer designed to eliminate suspicious or logically inconsistent profiles.

Examples include:

* Candidates claiming expert proficiency with zero practical experience.
* Impossible career timelines.
* Skill inflation and keyword stuffing.
* Contradictory experience histories.

This stage improves the robustness of the ranking pipeline and prevents adversarial profiles from appearing in the final shortlist.

---

### 3. Domain Classification using ConsultantBERT

The remaining candidates are evaluated using **ConsultantBERT**, our custom fine-tuned transformer model built on a distilled BERT architecture.

The model is trained specifically to distinguish between:

* IT Services / Consulting backgrounds
* Product Engineering backgrounds

This enables the ranking engine to prioritize candidates whose professional trajectory aligns more closely with product engineering requirements.

---

### 4. Composite Signal Weighting

Finally, a weighted scoring mechanism combines:

* Semantic similarity score
* Product-domain probability from ConsultantBERT
* Recruiter response probability
* Additional deterministic signals

The combined score is used to generate the final **Top 100 ranked candidates**, along with concise reasoning explaining the ranking.

---

## Offline Training Pipeline

The code used for:

* Data scraping and preprocessing
* Candidate labeling
* ConsultantBERT fine-tuning
* Embedding generation
* FAISS index creation

is available in:

```text
/offline_training/offline_pipeline.ipynb
```

This notebook contains the complete offline training and indexing workflow used to prepare the artifacts required during inference.

---

## Setup and Reproduction Instructions

Due to GitHub's file size restrictions (100 MB per file), the pre-computed FAISS index and ConsultantBERT model weights are hosted externally.

### Step 1: Download Pre-computed Artifacts

Download the required files from the public Google Drive folder:

**Google Drive Link:**
[Link](https://drive.google.com/drive/folders/1N2ImPewxUVoKemlJTyqLlRg9r8WrTMQ6?usp=drive_link)

After downloading, place the files in the root directory of this repository.

The directory structure should look as follows:

```text
├── rank.py
├── requirements.txt
├── candidates.jsonl                 # Hackathon Organizer Dataset
├── candidates.index                 # Downloaded FAISS Index
├── id_mapping.json                  # Downloaded ID Mapping
└── final_consultant_bert/           # Downloaded Model Directory
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

---

### Step 2: Install Dependencies

Ensure Python **3.11 or above** is installed.

Install all required dependencies:

```bash
pip install -r requirements.txt
```

All dependencies are CPU-compatible and optimized for the hackathon constraints.

---

### Step 3: Run the Ranking Pipeline

Execute the ranking script:

```bash
python rank.py
```

The script automatically loads:

* Candidate dataset
* FAISS index
* ConsultantBERT model
* ID mapping

and processes the candidates through the complete ranking funnel.

---

## Output

Upon successful execution, the script generates:

```text
team_insomniaks.csv
```

The output file contains:

* Candidate ID
* Final Rank
* Composite Score
* Supporting Ranking Reasoning

representing the **Top 100 ranked candidates** for the supplied Job Description.

---

## Hackathon Constraints Addressed

Our solution is specifically designed to satisfy the Redrob Hackathon constraints:

* Processes 100,000 candidates efficiently.
* Runs entirely on CPU.
* Uses less than 16 GB RAM.
* Requires no internet access during execution.
* Completes within the prescribed execution time.
* Robust against honeypots, keyword stuffing, and inconsistent profiles.

---

## Team

**Team Name:** Insomniaks

Thank you for reviewing our submission.
