import os
import json
import torch
import csv
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("\n" + "="*50)
print("🏆 STAGE 2 RANKING ENGINE (DOCKER/CPU ENVIRONMENT)")
print("="*50)

# 2. Paths & Stage 2 Weights
work_dir = './'
index_path = os.path.join(work_dir, "candidates.index")
mapping_path = os.path.join(work_dir, "id_mapping.json")
models_dir = os.path.join(work_dir, 'final_consultant_bert')

W_SEMANTIC = 0.40
W_PRODUCT = 0.30
W_EXPERIENCE = 0.20
W_BEHAVIORAL = 0.10

# Force CPU execution to exactly match the Hackathon constraints
device = torch.device("cpu")

print("⏳ 1/3: Loading precomputed indexes and models into memory...")
# Load precomputed FAISS and Metadata
index = faiss.read_index(index_path)
with open(mapping_path, "r") as f:
    data = json.load(f)
    id_order = data['order']
    metadata = data['metadata']

# Load Models
semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
tokenizer = AutoTokenizer.from_pretrained(models_dir)
product_model = AutoModelForSequenceClassification.from_pretrained(models_dir).to(device)
product_model.eval()

# Job Description
jd_text = """
Senior AI Engineer. Production experience with embeddings-based retrieval systems, 
vector databases (Pinecone, FAISS), and strong Python. 
Needs evaluation framework experience (NDCG, MRR).
Product company experience required.
"""

print("\n🚀 2/3: Starting Execution Timer!")
start_time = time.time()

# --- PHASE 1: MICROSECOND FAISS RETRIEVAL ---
# Encode JD and search precomputed index
jd_embedding = semantic_model.encode([jd_text], convert_to_numpy=True)
faiss.normalize_L2(jd_embedding)
semantic_scores, semantic_indices = index.search(jd_embedding, 1000) # Only pull Top 1000

# --- PHASE 2: COMPOSITE SCORING (Top 1000 Only) ---
print(f"Deep analyzing Top 1000 candidates using Consultant-BERT...")
final_candidates = []

for rank, idx in enumerate(semantic_indices[0]):
    # Get original ID and pre-parsed metadata
    cid = id_order[idx]
    cand_data = metadata[cid]
    
    # 1. Semantic Score (Cosine Similarity)
    sem_score_norm = min(max(semantic_scores[0][rank], 0.0), 1.0)
    
    # 2. Product-Fit Score (Consultant-BERT)
    inputs = tokenizer(cand_data['text'], return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = product_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        product_score = probs[0][1].item() # Prob of Product Engineer
    
    # 3. Experience Score (Cap at 7 years)
    exp_score = min(cand_data['exp_years'] / 7.0, 1.0)
    
    # 4. Behavioral Score
    beh_score = cand_data['response_rate']
    
    # THE MASTER COMPOSITE FORMULA
    composite = (W_SEMANTIC * sem_score_norm) + \
                (W_PRODUCT * product_score) + \
                (W_EXPERIENCE * exp_score) + \
                (W_BEHAVIORAL * beh_score)
                
    final_candidates.append({
        'candidate_id': cid,
        'composite_score': composite,
        'semantic': sem_score_norm,
        'product': product_score,
        'exp': exp_score,
        'beh': beh_score
    })

# --- PHASE 3: EXPORT (COMPLIANT FORMAT) ---
# FIX: Primary sort = Rounded score (Descending), Secondary sort = Candidate ID (Ascending)
final_candidates.sort(key=lambda x: (-round(x['composite_score'], 4), x['candidate_id']))

# Using a standard team name format for the validation script
output_file = os.path.join(work_dir, "team_insomniaks.csv")
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # EXACT HEADER MATCH REQUIRED BY JUDGES
    writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    
    for current_rank, row in enumerate(final_candidates[:100], start=1):
        # We inject the actual model metrics into the reasoning column to show our math to the judges
        reasoning_text = f"Semantic fit: {row['semantic']:.2f}, Product fit: {row['product']:.2f}, Exp score: {row['exp']:.2f}, Behavioral: {row['beh']:.2f}"
        
        writer.writerow([
            row['candidate_id'], 
            current_rank, 
            round(row['composite_score'], 4), 
            reasoning_text
        ])

execution_time = time.time() - start_time

print("\n" + "="*50)
print(f"✅ RANKING COMPLETE")
print(f"⏱️ Total Execution Time: {execution_time:.2f} seconds")
print(f"💾 Compliant CSV saved to: {output_file}")
print("="*50)

if execution_time < 300:
    print("🏆 SUCCESS: Execution time is safely under the 5-minute CPU constraint!")
else:
    print("⚠️ WARNING: Execution time exceeded 5 minutes.")