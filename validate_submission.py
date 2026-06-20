import os
import pandas as pd

def validate_generated_submission(csv_path: str):
    print(f"[Validator] Starting structural audit on: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Submission file missing at path: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 1. Assert precise schema constraints
    expected_columns = ["rank", "candidate_id", "score", "reasoning"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Schema mismatch! Expected {expected_columns}, got {list(df.columns)}")
        
    # 2. Check row completeness index limits
    if len(df) == 0:
        raise ValueError("Audit failed: The shortlist file contains 0 candidate records.")
        
    # 3. Assert strict score monotonicity (Non-increasing sorting order)
    scores = df["score"].tolist()
    is_monotonic_descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    if not is_monotonic_descending:
        raise ValueError("Audit failed: Candidate score vectors are not strictly non-increasing!")
        
    print(f"[Success] Validation passed! Code pipeline is fully compliant with rule indices.")

if __name__ == "__main__":
    validate_generated_submission("submission.csv")