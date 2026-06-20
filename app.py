import os
import streamlit as st
import pandas as pd
import orjson
import shutil
from run_pipeline import run_system_sorting_pipeline

# 1. Establish page layout parameters
st.set_page_config(
    page_title="Redrob AI Candidate Sorting Engine Sandbox",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Industrial-Grade Candidate Sorting Engine Sandbox")
st.subheader("Dual-Stage Hybrid Search & Listwise Rank Learning Platform [cite: 1]")
st.markdown(
    """
    This sandbox provides a fully isolated execution workspace to test high-throughput candidate sorting logic.
    Upload a custom candidate `.jsonl` stream file to run chronological validation and listwise re-ranking rules. [cite: 208]
    """
)

# 2. Add an sidebar displaying resource constraint parameters
st.sidebar.header("Operational Sandbox Limits [cite: 8]")
st.sidebar.markdown(
    """
    * **Hardware Bound:** CPU-only execution [cite: 34]
    * **Memory Limit:** Strictly capped at 16 GB RAM [cite: 34]
    * **Time Threshold:** Under 5.0 minutes for 100k records [cite: 34]
    * **Network State:** Isolated Offline Sandbox [cite: 34]
    """
)

# 3. Create file ingestion drag-and-drop workspace block
uploaded_file = st.file_uploader("Upload Test Candidate Stream File (.jsonl)", type=["jsonl"])

if uploaded_file is not None:
    st.info("[Sandbox] Ingesting uploaded data stream into local isolation folder...")
    
    # Establish local data track path directories
    os.makedirs("data", exist_ok=True)
    temp_jsonl_path = "data/uploaded_candidates.jsonl"
    output_csv_path = "submission.csv"
    
    # Write uploaded payload memory chunks natively to disk layer
    with open(temp_jsonl_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success("[Sandbox] Stream file successfully cached to local disk space.")
    
    # 4. Trigger core processing engine pipelines
    if st.button("🚀 Execute Multi-Stage Ranker Pipeline"):
        with st.spinner("Processing candidate vector matrices and executing mathematical validations..."):
            try:
                # Trigger the exact pipeline architecture used by your CI environment
                run_system_sorting_pipeline(
                    jsonl_input_path=temp_jsonl_path,
                    output_csv_path=output_csv_path
                )
                
                st.balloons()
                st.success("✨ Candidate sorting pipeline completed successfully!")
                
                # 5. Render output candidate shortlist datatables on screens
                if os.path.exists(output_csv_path):
                    df_results = pd.read_csv(output_csv_path)
                    
                    st.write("### 📋 Final Compliant Candidate Shortlist (Top 100 Profiles) [cite: 215]")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # 6. Offer a clean file download button interface element
                    with open(output_csv_path, "rb") as file_bytes:
                        st.download_button(
                            label="📥 Download Sorted List Sheet (submission.csv)",
                            data=file_bytes,
                            file_name="submission.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"[Pipeline Failure] Operational runtime crash: {str(e)}")