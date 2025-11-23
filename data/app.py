"""
Traffic Accident Auditor - Streamlit Application

Main entry point for the Hackathon project.
Audits consistency between a Written Incident Report and CCTV Video Footage.
"""

import streamlit as st
import tempfile
import cv2
import os
import pandas as pd

# --- IMPORT YOUR MODULES ---
# Ensure you have text_parser.py, video_analyzer.py, and scoring.py in the same folder
from text_parser import ReportProcessor
from video_analyzer import process_video_sequence 
from scoring import score_consistency

def get_video_metadata(file_path):
    """Helper to get real FPS and Frame Count from the uploaded file."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return 30.0, 100 # Default fallback
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

def get_specific_frame(file_path, frame_number):
    """Extracts a specific frame image to show as evidence."""
    cap = cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def main():
    st.set_page_config(
        page_title="Traffic Incident Auditor",
        page_icon="🚔",
        layout="wide"
    )
    
    # --- HEADER ---
    st.title("🚔 Traffic Incident Auditor")
    st.markdown("""
    **Automated Consistency Check:** Compare an Officer's Incident Report against CADP CCTV Footage.
    
    **Auditing Targets:**
    1.  🕒 **Time Consistency:** Did the crash happen when the report says it did?
    2.  💥 **Severity Match:** Does the video damage match the reported severity?
    """)
    
    st.divider()
    
    # --- INPUT SECTION ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload CCTV Footage")
        video_file = st.file_uploader(
            "Upload Crash Video (MP4/AVI)",
            type=["mp4", "mov", "avi", "mkv"]
        )

    with col2:
        st.subheader("2. Enter Incident Report")
        text_desc = st.text_area(
            "Paste Text Report",
            height=150,
            placeholder="Example: 'A Severe head-on collision occurred at 00:02:35 involving two vehicles.'",
            help="Ensure the text contains a time (HH:MM:SS) and a severity keyword (Minor, Moderate, Severe)."
        )
    
    st.divider()
    
    # --- ANALYSIS BUTTON ---
    if st.button("🔍 Run Consistency Audit", type="primary", use_container_width=True):
        if not video_file or not text_desc.strip():
            st.warning("⚠️ Please provide both a video file and a text report.")
            return
        
        # 1. SETUP TEMP FILE
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name
        except Exception as e:
            st.error(f"File Error: {e}")
            return

        # --- PIPELINE EXECUTION ---
        
        # STAGE 1: VIDEO ANALYSIS (CV)
        with st.spinner("📹 Analyzing CADP Frames for Impact..."):
            try:
                # Get real metadata to make the simulation/model accurate
                real_fps, real_total_frames = get_video_metadata(tmp_path)
                
                # CALL YOUR VIDEO MODULE
                # Note: We pass the filename "uploaded_video" just for the log
                cv_results = process_video_sequence("uploaded_video", real_total_frames, real_fps)
                
            except Exception as e:
                st.error(f"❌ Video Analysis Failed: {e}")
                os.unlink(tmp_path)
                return

        # STAGE 2: TEXT ANALYSIS (NLP)
        with st.spinner("📝 Parsing Incident Report..."):
            try:
                # Initialize your class
                nlp_processor = ReportProcessor()
                
                # CALL YOUR NLP MODULE
                nlp_results = nlp_processor.process_report(text_desc)
                
            except Exception as e:
                st.error(f"❌ Text Parsing Failed: {e}")
                os.unlink(tmp_path)
                return

        # STAGE 3: CONSISTENCY SCORING
        with st.spinner("⚖️ Calculating Audit Score..."):
            try:
                # CALL YOUR SCORING MODULE
                final_output = score_consistency(nlp_results, cv_results)
                
            except Exception as e:
                st.error(f"❌ Scoring Failed: {e}")
                os.unlink(tmp_path)
                return

        # --- RESULTS DISPLAY ---
        
        st.success("✅ Audit Complete!")
        
        # 1. TOP LEVEL SCORE
        score = final_output['score']
        score_color = "🟢" if score == 100 else "🔴"
        
        st.markdown(f"""
        <div style="text-align: center; font-size: 24px; padding: 10px; border: 2px solid #333; border-radius: 10px;">
            Audit Score: <b>{score}/100</b> {score_color}
        </div>
        <br>
        """, unsafe_allow_html=True)

        # 2. DETAILED COMPARISON COLUMNS
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.info("📄 **Report Claims**")
            st.metric("Reported Time", f"{nlp_results.get('TReport', 'N/A')} sec")
            st.metric("Reported Severity", nlp_results.get('SeverityReport', 'N/A'))

        with res_col2:
            st.warning("📹 **Video Evidence**")
            st.metric("Actual Impact Time", f"{cv_results.get('T_Actual', 'N/A')} sec")
            st.metric("AI Classified Severity", cv_results.get('Severity_Actual', 'N/A'))

        # 3. EVIDENCE TABLE
        st.subheader("📊 Detailed Findings")
        if final_output["details"]:
            df = pd.DataFrame(final_output["details"])
            # Renaming for cleaner UI
            df = df[["claim_type", "claim_value", "video_value", "result", "note"]]
            df.columns = ["Check Type", "Report Says", "Video Shows", "Status", "Notes"]
            st.dataframe(df, use_container_width=True, hide_index=True)

        # 4. VISUAL PROOF (The Frame)
        st.divider()
        st.subheader("📸 Visual Proof: Impact Frame")
        
        impact_frame_idx = cv_results.get("F_Actual", 0)
        
        # Extract the specific frame from the video
        proof_image = get_specific_frame(tmp_path, impact_frame_idx)
        
        if proof_image is not None:
            st.image(proof_image, caption=f"Detected Collision at Frame {impact_frame_idx}", use_container_width=True)
        else:
            st.warning("Could not extract impact frame.")

        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass

if __name__ == "__main__":
    main()