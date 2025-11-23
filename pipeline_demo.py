import json
from scoring import score_consistency

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
    
    # Call the function from scoring.py
    audit_report = score_consistency(nlp_output, cv_output)

    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n📊 FINAL AUDIT REPORT")
    print("="*40)
    
    # Extract details from the list structure returned by scoring.py
    details = audit_report['details']
    
    # Find specific checks for clean printing
    time_check = next((d for d in details if d['claim_type'] == "Time of Impact"), None)
    sev_check = next((d for d in details if d['claim_type'] == "Accident Severity"), None)
    
    # 1. TIME REPORT
    if time_check:
        print(f"TIME CHECK:      [{time_check['result']}]")
        print(f"   Values:       Report says {time_check['claim_value']} vs Video shows {time_check['video_value']}")
        print(f"   Note:         {time_check['note']}")
    else:
        print("TIME CHECK:      [SKIPPED/ERROR]")

    print("-" * 40)

    # 2. SEVERITY REPORT
    if sev_check:
        print(f"SEVERITY CHECK:  [{sev_check['result']}]")
        print(f"   Values:       Report says '{sev_check['claim_value']}' vs Video shows '{sev_check['video_value']}'")
        print(f"   Note:         {sev_check['note']}")
    else:
        print("SEVERITY CHECK:  [SKIPPED/ERROR]")

    print("="*40)
    
    # 3. OVERALL SCORE
    score = audit_report['score']
    print(f"🏆 CALCULATED SCORE: {score}/100")
    
    if score == 100:
        print("✅ RESULT: REPORT IS CONSISTENT")
    else:
        print("❌ RESULT: DISCREPANCY DETECTED")

    # Export Logic
    with open("final_audit_score.json", "w") as f:
        json.dump(audit_report, f, indent=4)
    print("\n📄 Full report exported to 'final_audit_score.json'")

if __name__ == "__main__":
    run_pipeline_test()