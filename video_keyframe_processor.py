import json
import random
import math
from typing import Dict, Any, Tuple

# --- Configuration & Heuristics (Based on Spec 2.2) ---

# Mock Frame Rate for the CADP footage
MOCK_FPS = 30.0

# Severity Classification Thresholds (Heuristic Proxy Metric: Normalized Bounding Box Area)
# These values simulate the empirical thresholds set during the sprint.
THRESHOLD_SEVERE = 0.15  # Area >= 15% of total frame area
THRESHOLD_MODERATE = 0.05 # Area >= 5% but < 15% of total frame area
# Anything below 5% is considered 'Minor'

# --- Simulation Functions ---

def simulate_yolo_detection(total_frames: int) -> Tuple[int, float]:
    """
    Simulates the YOLOv8-Nano Collision Detection task.

    In a real scenario, this function would iterate through each frame,
    run model inference (model.predict(frame)), and find the first frame
    where the 'collision_event' class exceeds the high-confidence threshold (0.85).

    :param total_frames: Total number of frames in the sequence.
    :return: A tuple of (F_Actual, Normalized_Impact_Area)
    """
    print(f"--- Simulating YOLOv8-Nano Collision Detection on {total_frames} frames ---")

    # 1. Simulate finding the first high-confidence collision frame (F_Actual)
    # The collision is simulated to happen between 10% and 90% of the video duration.
    F_Actual = random.randint(math.ceil(total_frames * 0.1), math.floor(total_frames * 0.9))
    print(f"-> Model detected 'collision_event' (Conf > 0.85) first at Frame: {F_Actual}")

    # 2. Simulate calculating the Normalized Impact Area
    # This is a proxy for the severity (larger area = more damage/debris).
    # Area is simulated to be between 1% and 25% of the total frame area.
    Normalized_Area = random.uniform(0.01, 0.25)
    print(f"-> Detected Bounding Box Normalized Area (Proxy Metric): {Normalized_Area:.4f}")

    return F_Actual, Normalized_Area

def classify_severity(normalized_area: float) -> str:
    """
    Classifies the severity based on the Heuristic Proxy Metric (Normalized Area).

    This simulates the Image Classification Model task by applying the
    pre-defined bounding box area thresholds.

    :param normalized_area: The calculated bounding box area normalized by total frame area.
    :return: The classified severity string ('Severe', 'Moderate', or 'Minor').
    """
    if normalized_area >= THRESHOLD_SEVERE:
        severity = "Severe"
    elif normalized_area >= THRESHOLD_MODERATE:
        severity = "Moderate"
    else:
        severity = "Minor"

    return severity

def process_video_sequence(sequence_name: str, total_frames: int, fps: float) -> Dict[str, Any]:
    """
    The main processor function for Section 2.2.

    It orchestrates the collision detection and severity classification.

    :param sequence_name: Identifier for the video sequence being processed.
    :param total_frames: The total number of frames in the video sequence.
    :param fps: The frames per second of the video sequence.
    :return: A dictionary containing the final extracted variables (TActual, SeverityActual).
    """
    print(f"\nProcessing Video Sequence: {sequence_name} (FPS: {fps})")

    # --- Task 1: Collision Detection (Determine F_Actual and T_Actual) ---

    # Simulates the model running inference on keyframes
    F_Actual, normalized_area = simulate_yolo_detection(total_frames)

    # Calculate T_Actual (Frame timestamp converted to seconds)
    T_Actual = F_Actual / fps

    # --- Task 2: Severity Classification (Determine Severity_Actual) ---

    Severity_Actual = classify_severity(normalized_area)
    print(f"-> Classified Severity (based on area proxy): {Severity_Actual}")

    # --- Task 3: Output Compilation ---

    output_data = {
        "Video_Sequence_ID": sequence_name,
        "Collision_Detected": True,
        "F_Actual": F_Actual,
        "T_Actual": round(T_Actual, 3), # Round to 3 decimal places for clean output
        "Severity_Actual": Severity_Actual
    }

    return output_data

def export_to_json(data: Dict[str, Any], filename: str = "video_processor_output.json"):
    """
    Simulates the export to Sheets/JSON step for the Consistency Auditor.

    :param data: The data record to export.
    :param filename: The file path to save the JSON.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n✅ Output successfully exported to {filename} for the Consistency Auditor.")


# --- Main Execution Block (Demo) ---

if __name__ == "__main__":
    # Example 1: A short sequence with a high-severity collision
    SEQUENCE_1_NAME = "Demo_CADP_Sequence_A_HighImpact"
    TOTAL_FRAMES_1 = 900 # A 30-second clip (900/30)

    # Run the processor
    result_1 = process_video_sequence(SEQUENCE_1_NAME, TOTAL_FRAMES_1, MOCK_FPS)

    # Export to the next stage
    export_to_json(result_1, "video_output_A.json")

    print("\n" + "="*50 + "\n")

    # Example 2: A longer sequence with a different (simulated) severity
    SEQUENCE_2_NAME = "Demo_CADP_Sequence_B_Longer"
    TOTAL_FRAMES_2 = 3600 # A 120-second clip (3600/30)

    # Run the processor
    result_2 = process_video_sequence(SEQUENCE_2_NAME, TOTAL_FRAMES_2, MOCK_FPS)

    # Export to the next stage
    export_to_json(result_2, "video_output_B.json")

    print("\n--- Final Extracted Data for Consistency Auditor ---")
    print("Sequence A:", json.dumps(result_1, indent=4))
    print("Sequence B:", json.dumps(result_2, indent=4))