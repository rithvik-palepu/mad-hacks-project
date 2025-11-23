import json
from scoring import ConsistencyAuditor

# --- MOCKING THE IMPORTS FROM YOUR OTHER FILES ---
# In your real project, you would import these:
# from text_parser import ReportProcessor
# from video_analyzer import process_video_sequence

def run_pipeline_test():
    print("🚀 STARTING 3-STAGE PIPELINE TEST\n")
    
    # ---------------------------------------------------------
    # STAGE 1: NLP (Text Input Processor)
    # ---------------------------------------------------------
    print("1️⃣  Running NLP Processor...")
    # Simulating the output of your ReportProcessor
    nlp_output = {
        "TReport": 152,  # 2 minutes 32 seconds
        "SeverityReport": "Severe",
        "RawTextSnippet": "Subject vehicle collision occurred at 00:02:32..."
    }
    print(f"   -> Extracted Time: {nlp_output['TReport']}s")
    print(f"   -> Extracted Severity: {nlp_output['SeverityReport']}")

    # ---------------------------------------------------------
    # STAGE 2: CV (Video Input Processor)
    # ---------------------------------------------------------
    print("\n2️⃣  Running Video Processor...")
    # Simulating the output of your process_video_sequence
    # Scenario: The video shows the crash actually happened at 155s (3s diff)
    cv_output = {
        "Video_Sequence_ID": "CADP_Clip_001",
        "Collision_Detected": True,
        "F_Actual": 4650, # Frame number
        "T_Actual": 155.0, # Seconds
        "Severity_Actual": "Severe"
    }
    print(f"   -> Detected Time: {cv_output['T_Actual']}s")
    print(f"   -> Classified Severity: {cv_output['Severity_Actual']}")

    # ---------------------------------------------------------
    # STAGE 3: CONSISTENCY AUDITOR (Scoring)
    # ---------------------------------------------------------
    print("\n3️⃣  Running Consistency Auditor...")
    
    auditor = ConsistencyAuditor(time_threshold=5.0)
    audit_report = auditor.audit(nlp_output, cv_output)

    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n📊 FINAL AUDIT REPORT")
    print("="*30)
    
    time_res = audit_report['Time_Audit']
    sev_res = audit_report['Severity_Audit']
    
    print(f"TIME CHECK:      [{time_res['Consistency']}]")
    print(f"   Difference:   {time_res['Discrepancy_Seconds']} seconds")
    print(f"   (Report: {time_res['Reported']}s vs Video: {time_res['Actual']}s)")
    print("-" * 30)
    print(f"SEVERITY CHECK:  [{sev_res['Match']}]")
    print(f"   (Report: '{sev_res['Reported']}' vs Video: '{sev_res['Actual']}')")
    print("="*30)
    
    if audit_report["Overall_Status"] == "PASS":
        print("✅ RESULT: REPORT IS CONSISTENT")
    else:
        print("❌ RESULT: DISCREPANCY DETECTED")

    # Export Logic
    with open("final_audit_score.json", "w") as f:
        json.dump(audit_report, f, indent=4)
    print("\n📄 Full report exported to 'final_audit_score.json'")

if __name__ == "__main__":
    run_pipeline_test()