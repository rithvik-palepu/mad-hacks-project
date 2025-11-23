"""
Consistency Auditor Module (Part 2.3)

This module compares extracted text data (NLP) against video analysis data (CV).
It calculates a consistency score based on Time Difference and Severity Matching.
"""

from typing import Dict, Any, List

# --- CONFIGURATION ---
TIME_THRESHOLD_SECONDS = 5  # Allowable margin of error for timestamp
SCORE_WEIGHT_TIME = 50      # Points awarded for matching time
SCORE_WEIGHT_SEVERITY = 50  # Points awarded for matching severity

def score_consistency(nlp_data: Dict[str, Any], cv_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Audits the consistency between the Written Report and Video Evidence.

    Args:
        nlp_data: Dictionary containing 'TReport' (int) and 'SeverityReport' (str).
        cv_data: Dictionary containing 'T_Actual' (int), 'Severity_Actual' (str), 
                 and 'Collision_Detected' (bool).

    Returns:
        Dict containing:
            - 'score': Total score (0-100)
            - 'details': List of dictionaries for the UI table
            - 'status': Overall status string
    """
    
    details = []
    total_score = 0
    
    # --- 1. PRE-FLIGHT CHECK: DID A COLLISION HAPPEN? ---
    # If the CV model didn't detect a crash, we cannot verify time or severity.
    collision_detected = cv_data.get("Collision_Detected", False)
    
    if not collision_detected:
        return {
            "score": 0,
            "status": "NO COLLISION DETECTED IN VIDEO",
            "details": [{
                "claim_type": "Event Existence",
                "claim_value": "Accident Reported",
                "video_value": "No Collision Detected",
                "result": "FAIL",
                "note": "The AI could not find a crash in the footage."
            }]
        }

    # --- 2. AUDIT: TIME CONSISTENCY ---
    t_report = nlp_data.get("TReport", -1)
    t_actual = cv_data.get("T_Actual", -1)
    time_status = "FAIL"
    time_note = ""
    
    # Handle missing data cases
    if t_report == -1:
        time_note = "No time found in text report."
        time_status = "MISSING DATA"
    elif t_actual == -1:
        time_note = "No impact timestamp calculated."
        time_status = "MISSING DATA"
    else:
        # THE CORE LOGIC: Absolute Difference
        delta_t = abs(t_report - t_actual)
        
        if delta_t <= TIME_THRESHOLD_SECONDS:
            total_score += SCORE_WEIGHT_TIME
            time_status = "MATCH"
            time_note = f"Difference of {delta_t}s is within tolerance ({TIME_THRESHOLD_SECONDS}s)."
        else:
            time_status = "INCONSISTENT"
            time_note = f"Time gap ({delta_t}s) exceeds threshold."

    # Log Time Detail
    details.append({
        "claim_type": "Time of Impact",
        "claim_value": f"{t_report} sec" if t_report != -1 else "Unknown",
        "video_value": f"{t_actual} sec" if t_actual != -1 else "Unknown",
        "result": time_status,
        "note": time_note
    })

    # --- 3. AUDIT: SEVERITY CONSISTENCY ---
    sev_report = str(nlp_data.get("SeverityReport", "Unknown")).lower().strip()
    sev_actual = str(cv_data.get("Severity_Actual", "Unknown")).lower().strip()
    sev_status = "FAIL"
    sev_note = ""

    if sev_report == "unknown" or sev_actual == "unknown":
        sev_status = "MISSING DATA"
        sev_note = "Could not determine severity from one or both sources."
    elif sev_report == sev_actual:
        total_score += SCORE_WEIGHT_SEVERITY
        sev_status = "MATCH"
        sev_note = "Reported severity matches video analysis."
    else:
        # Simple Logic: Mismatch
        sev_status = "MISMATCH"
        sev_note = f"Report says '{sev_report}', Video shows '{sev_actual}'."

    # Log Severity Detail
    details.append({
        "claim_type": "Accident Severity",
        "claim_value": sev_report.title(),
        "video_value": sev_actual.title(),
        "result": sev_status,
        "note": sev_note
    })

    # --- 4. FINAL AGGREGATION ---
    return {
        "score": total_score,
        "status": "COMPLETE",
        "details": details
    }