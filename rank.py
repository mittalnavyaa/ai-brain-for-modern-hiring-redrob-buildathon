import json
import faiss
import csv
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

work_dir = '.'

print("1. Loading Pre-computed Artifacts...")

# Load FAISS
index = faiss.read_index(f"{work_dir}/candidates.index")
with open(f"{work_dir}/id_mapping.json", "r") as f:
    id_mapping = json.load(f)

# Load Models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained(f"{work_dir}/final_consultant_bert")
model = AutoModelForSequenceClassification.from_pretrained(f"{work_dir}/final_consultant_bert")
model.eval() # Set model to evaluation mode

print("2. Retrieving Top 500 Semantic Matches from FAISS...")
# The Job Description Summary
jd_text = """
Senior AI Engineer. Production experience with embeddings-based retrieval systems,
vector databases (Pinecone, FAISS), and strong Python.
Needs evaluation framework experience (NDCG, MRR).
Must have product company experience, not pure consulting or pure academic research.
"""

jd_embedding = embedder.encode([jd_text], convert_to_numpy=True)
faiss.normalize_L2(jd_embedding)

# Search FAISS
distances, indices = index.search(jd_embedding, 500)
# Map exact FAISS distances to candidate IDs for accurate scoring
semantic_scores = {id_mapping[idx]: dist for dist, idx in zip(distances[0], indices[0])}
top_500_ids = set(semantic_scores.keys())

print("3. Extracting Full Profiles for Top 500...")
candidates_data = []
with open(f"{work_dir}/candidates.jsonl", "rt", encoding="utf-8") as f:
    for line in f:
        cand = json.loads(line)
        if cand['candidate_id'] in top_500_ids:
            candidates_data.append(cand)

print("4. Running the Gauntlet (Honeypots, BERT, and Signals)...")
final_rankings = []

for cand in candidates_data:
    cid = cand['candidate_id']
    profile = cand.get('profile', {})
    signals = cand.get('redrob_signals', {})

    # --- A. HONEYPOT & TRAP CHECK ---
    # Trap: "Expert" proficiency but 0 duration.
    is_honeypot = False
    for skill in cand.get('skills', []):
        if skill.get('proficiency') in ['advanced', 'expert'] and skill.get('duration_months', 1) == 0:
            is_honeypot = True
            break

    if is_honeypot:
        continue # Instantly drop them from the ranking

    # --- B. CONSULTANT-BERT CLASSIFICATION ---
    text = f"{profile.get('headline', '')} {profile.get('summary', '')}"
    inputs = tokenizer(text[:512], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        product_score = probs[0][1].item() # Probability of being Label 1 (Product Engineer)

    # --- C. BEHAVIORAL SIGNALS ---
    # In the real world, inactive/non-responsive candidates are useless.
    response_rate = signals.get('recruiter_response_rate', 0.5)

    # --- D. FINAL COMPOSITE SCORE ---
    # 40% Semantic Match, 40% Product-Domain Fit, 20% Recruiter Response
    semantic_score = semantic_scores.get(cid, 0.5)

    composite_score = (semantic_score * 0.4) + (product_score * 0.4) + (response_rate * 0.2)

    # --- E. GENERATE DYNAMIC REASONING ---
    exp_years = profile.get('years_of_experience', 0)
    reasoning = f"Matched with {exp_years} yrs exp. Product-fit probability {product_score:.0%}. Behavioral response rate {response_rate:.0%}."

    final_rankings.append({
        "candidate_id": cid,
        "score": round(composite_score, 4),
        "reasoning": reasoning
    })

print("5. Sorting and Formatting Submission...")
# Sort highest score first, break ties using candidate_id ascending
final_rankings.sort(key=lambda x: (-x['score'], x['candidate_id']))

# Extract exactly top 100
top_100 = final_rankings[:100]

# Write to CSV!
TEAM_ID = "team_insomniaks"
csv_path = f"{work_dir}/{TEAM_ID}.csv"

with open(csv_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, cand in enumerate(top_100, start=1):
        writer.writerow([cand['candidate_id'], rank, cand['score'], cand['reasoning']])

print(f"\nSubmission file generated successfully at: {csv_path}")