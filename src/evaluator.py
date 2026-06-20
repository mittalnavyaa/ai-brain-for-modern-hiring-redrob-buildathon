import numpy as np
from typing import List, Dict, Any, Tuple

def calculate_cumulative_career_length(intervals: List[Tuple[float, float]]) -> float:
    """
    Computes the Lebesgue measure (true non-overlapping duration) of an interval union.
    Prevents candidates from masking gaps by claiming overlapping full-time positions.
    """
    if not intervals:
        return 0.0
    
    # Sort intervals by their start year
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    union_intervals = []
    
    curr_start, curr_end = sorted_intervals[0]
    for next_start, next_end in sorted_intervals[1:]:
        if next_start <= curr_end:
            # Overlap detected; extend the current interval bound
            curr_end = max(curr_end, next_end)
        else:
            # Disjoint interval; push current tracker and reset
            union_intervals.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    union_intervals.append((curr_start, curr_end))
    
    # Sum up the unique non-overlapping durations
    return sum(end - start for start, end in union_intervals)

def verify_chronology(experiences: List[Dict[str, Any]], company_ages: Dict[str, float]) -> float:
    """
    Runs deterministic timeline checks over work and academic history.
    Returns 0.0 (hard-drop) if impossible structural traps or honeypots trigger.
    """
    if not experiences:
        return 1.0
        
    intervals = []
    raw_duration_sum = 0.0
    delta = 1.0  # Pre-incorporation error margin
    
    for exp in experiences:
        s_i = float(exp.get("start_year", 0))
        o_i = float(exp.get("end_year", 0))
        c_i = str(exp.get("company_id", ""))
        is_part_time = exp.get("is_part_time", False) or exp.get("is_freelance", False)
        
        # 1. Internal Timeline Contradiction check
        if o_i < s_i:
            return 0.0
            
        # 2. Company Age Anachronism check
        firm_lifetime = company_ages.get(c_i, 99.0)  # Default to high number if unknown
        if (o_i - s_i) > (firm_lifetime + delta):
            return 0.0
            
        # Track intervals for non-part-time/freelance engagements
        if not is_part_time:
            intervals.append((s_i, o_i))
            raw_duration_sum += (o_i - s_i)
            
    # 3. Interval Union Check for Job Hoarder/Overlapping Traps
    if intervals:
        true_career_len = calculate_cumulative_career_length(intervals)
        if true_career_len > 0 and (raw_duration_sum / true_career_len) > 1.5:
            return 0.0  # Chronology manipulation anomaly detected
            
    return 1.0

def evaluate_title_divergence(title_vector: np.ndarray, target_vector: np.ndarray, current_title_raw: str) -> float:
    """
    Calculates cosine similarity on CPU and checks for keyword stuffing.
    Applies severe penalties if high-level domains are claimed by unrelated job roles.
    """
    # Fast unit vector dot-product computation
    norm_title = np.linalg.norm(title_vector)
    norm_target = np.linalg.norm(target_vector)
    
    if norm_title == 0 or norm_target == 0:
        similarity = 0.0
    else:
        similarity = float(np.dot(title_vector, target_vector) / (norm_title * norm_target))
        
    # Check for keyword-stuffer traps (e.g., HR Manager listing complex vector search tags)
    blacklist_terms = {"manager", "recruiter", "coordinator", "sales", "hr"}
    words = set(current_title_raw.lower().replace("-", " ").replace("/", " ").split())
    
    if words.intersection(blacklist_terms) and similarity < 0.45:
        return 0.05  # Severe penalty multiplier to drop fake profiles
        
    return max(similarity, 0.10)