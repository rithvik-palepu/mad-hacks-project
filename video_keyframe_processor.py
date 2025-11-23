"""
Video Keyframe Processor for Traffic Accident Analysis
Extracts keyframes from video, detects collisions, and classifies severity
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
from ultralytics import YOLO
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image


class VideoKeyframeProcessor:
    """
    Processes video files to detect traffic accidents and classify severity.
    """
    
    def __init__(
        self,
        collision_model_path: str = "yolov8n.pt",
        severity_model_path: Optional[str] = None,
        keyframe_interval: int = 5,
        collision_threshold: float = 0.5
    ):
        """
        Initialize the Video Keyframe Processor.
        
        Args:
            collision_model_path: Path to YOLOv8 model for collision detection
            severity_model_path: Path to custom severity classification model
            keyframe_interval: Extract keyframe every N frames
            collision_threshold: Confidence threshold for collision detection
        """
        self.keyframe_interval = keyframe_interval
        self.collision_threshold = collision_threshold
        
        # Load YOLOv8 for collision detection
        print(f"Loading collision detection model: {collision_model_path}")
        self.collision_detector = YOLO(collision_model_path)
        
        # Load/Initialize severity classifier
        self.severity_classifier = self._init_severity_classifier(severity_model_path)
        self.severity_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.severity_classes = ["Minor", "Moderate", "Severe"]
        
    def _init_severity_classifier(self, model_path: Optional[str]) -> nn.Module:
        """
        Initialize severity classification model.
        Uses pre-trained MobileNetV2 or loads custom model.
        """
        if model_path and Path(model_path).exists():
            print(f"Loading custom severity model: {model_path}")
            model = torch.load(model_path)
        else:
            print("Initializing pre-trained MobileNetV2 for severity classification")
            model = models.mobilenet_v2(pretrained=True)
            # Modify final layer for 3-class severity classification
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        
        model.eval()
        return model
    
    def extract_keyframes(
        self, 
        video_path: str, 
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract keyframes from video file.
        
        Args:
            video_path: Path to input video file
            output_dir: Optional directory to save keyframe images
            
        Returns:
            List of dictionaries containing keyframe info
        """
        print(f"\nProcessing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        keyframes = []
        frame_count = 0
        
        # Create output directory if specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keyframe at specified intervals
            if frame_count % self.keyframe_interval == 0:
                timestamp = frame_count / fps if fps > 0 else 0
                
                keyframe_info = {
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'frame': frame,
                    'image_path': None
                }
                
                # Save keyframe image if output directory specified
                if output_dir:
                    image_filename = f"keyframe_{frame_count:06d}.jpg"
                    image_path = Path(output_dir) / image_filename
                    cv2.imwrite(str(image_path), frame)
                    keyframe_info['image_path'] = str(image_path)
                
                keyframes.append(keyframe_info)
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(keyframes)} keyframes from {total_frames} total frames")
        
        return keyframes
    
    def detect_collision(self, frame: np.ndarray) -> Tuple[bool, float, List]:
        """
        Detect collision in a single frame using YOLOv8.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Tuple of (collision_detected, confidence, detections)
        """
        # Run inference
        results = self.collision_detector(frame, verbose=False)
        
        # Analyze detections for collision indicators
        collision_detected = False
        max_confidence = 0.0
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            # Look for vehicles (car, truck, bus, motorcycle)
            vehicle_classes = [2, 3, 5, 7]  # COCO class IDs for vehicles
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in vehicle_classes:
                    detections.append({
                        'class': cls,
                        'confidence': conf,
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    })
                    
                    # Check for collision indicators:
                    # 1. High-confidence vehicle detections
                    # 2. Overlapping bounding boxes (potential collision)
                    if conf > self.collision_threshold:
                        max_confidence = max(max_confidence, conf)
                        collision_detected = True
        
        # Enhanced collision detection: check for overlapping vehicles
        if len(detections) >= 2:
            collision_detected = self._check_vehicle_overlap(detections) or collision_detected
        
        return collision_detected, max_confidence, detections
    
    def _check_vehicle_overlap(self, detections: List[Dict]) -> bool:
        """
        Check if vehicle bounding boxes overlap significantly (collision indicator).
        """
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                bbox1 = detections[i]['bbox']
                bbox2 = detections[j]['bbox']
                
                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(bbox1, bbox2)
                
                # High IoU suggests collision/overlap
                if iou > 0.3:
                    return True
        
        return False
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def classify_severity(self, frame: np.ndarray, detections: List[Dict] = None) -> Tuple[str, float]:
        """
        Classify accident severity from frame using rule-based approach.
        
        Args:
            frame: Input frame (numpy array)
            detections: Vehicle detections from collision detection
            
        Returns:
            Tuple of (severity_class, confidence)
        """
        # Rule-based severity assessment for hackathon speed
        severity_score = 0
        confidence = 0.0
        
        if detections:
            # Factor 1: Number of vehicles involved
            num_vehicles = len(detections)
            if num_vehicles >= 3:
                severity_score += 30
            elif num_vehicles == 2:
                severity_score += 15
            
            # Factor 2: Maximum overlap (IoU) between vehicles
            max_iou = 0.0
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    iou = self._calculate_iou(
                        detections[i]['bbox'], 
                        detections[j]['bbox']
                    )
                    max_iou = max(max_iou, iou)
            
            # High overlap = severe impact
            if max_iou > 0.5:
                severity_score += 40
            elif max_iou > 0.3:
                severity_score += 25
            elif max_iou > 0.15:
                severity_score += 10
            
            # Factor 3: Size of vehicles (larger bbox = more debris/damage)
            total_area = 0
            for det in detections:
                bbox = det['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                total_area += area
            
            # Normalize by frame size
            frame_area = frame.shape[0] * frame.shape[1]
            area_ratio = total_area / frame_area
            
            if area_ratio > 0.4:
                severity_score += 30
            elif area_ratio > 0.25:
                severity_score += 15
            
            # Calculate confidence based on detection quality
            avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            confidence = min(0.95, avg_conf + 0.2)  # Boost confidence for rule-based
        
        # Map score to severity class
        if severity_score >= 60:
            severity_class = "Severe"
        elif severity_score >= 30:
            severity_class = "Moderate"
        else:
            severity_class = "Minor"
        
        # If no detections, fall back to ML model
        if not detections:
            return self._ml_classify_severity(frame)
        
        return severity_class, confidence
    
    def _ml_classify_severity(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        ML-based severity classification (fallback method).
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        input_tensor = self.severity_transform(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.severity_classifier(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        severity_class = self.severity_classes[predicted.item()]
        confidence_score = confidence.item()
        
        return severity_class, confidence_score
    
    def process_video(
        self, 
        video_path: str,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Complete video processing pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Optional directory for saving keyframes
            
        Returns:
            Dictionary containing processing results
        """
        print("\n" + "="*60)
        print("VIDEO KEYFRAME PROCESSOR - STARTING ANALYSIS")
        print("="*60)
        
        # Extract keyframes
        keyframes = self.extract_keyframes(video_path, output_dir)
        
        if not keyframes:
            return {
                'collision_detected': False,
                'error': 'No keyframes extracted from video'
            }
        
        # Analyze each keyframe for collision
        collision_frame = None
        collision_timestamp = None
        collision_confidence = 0.0
        
        print(f"\nAnalyzing {len(keyframes)} keyframes for collision detection...")
        
        for idx, keyframe_info in enumerate(keyframes):
            frame = keyframe_info['frame']
            detected, confidence, detections = self.detect_collision(frame)
            
            if detected:
                # Keep track of highest confidence collision
                if confidence > collision_confidence:
                    collision_frame = keyframe_info['frame_number']
                    collision_timestamp = keyframe_info['timestamp']
                    collision_confidence = confidence

                    print(f"  Frame {keyframe_info['frame_number']}: "
                      f"COLLISION DETECTED (confidence: {confidence:.3f})")
        
        # If collision detected, classify severity
        severity_actual = None
        severity_confidence = 0.0
        collision_detections = None
        
        if collision_frame is not None:
            print(f"\nCollision detected at frame {collision_frame} "
                  f"(timestamp: {collision_timestamp:.2f}s)")
            
            # Find the collision keyframe and re-run detection for severity analysis
            collision_keyframe = next(
                kf for kf in keyframes if kf['frame_number'] == collision_frame
            )
            
            # Get detections for severity classification
            _, _, collision_detections = self.detect_collision(collision_keyframe['frame'])
            
            # Classify severity with detection context
            severity_actual, severity_confidence = self.classify_severity(
                collision_keyframe['frame'],
                detections=collision_detections
            )
            
            print(f"Severity classification: {severity_actual} "
                  f"(confidence: {severity_confidence:.3f})")
        
        # Prepare output
        result = {
            'video_path': video_path,
            'collision_detected': collision_frame is not None,
            'T_actual': collision_timestamp,
            'collision_frame': collision_frame,
            'collision_confidence': collision_confidence,
            'severity_actual': severity_actual,
            'severity_confidence': severity_confidence,
            'total_keyframes': len(keyframes),
            'keyframe_interval': self.keyframe_interval
        }
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results: {json.dumps(result, indent=2)}")
        
        return result
    
    def export_results(self, results: Dict, output_path: str):
        """
        Export results to JSON file.
        
        Args:
            results: Processing results dictionary
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults exported to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = VideoKeyframeProcessor(
        collision_model_path="yolov8n.pt",  # Will download if not present
        keyframe_interval=5,  # Extract every 5th frame
        collision_threshold=0.5
    )
    
    # Process video
    video_path = "data/clip1/clip1.mp4"
    results = processor.process_video(
        video_path=video_path,
        output_dir="output/keyframes"
    )
    
    # Export results
    processor.export_results(results, "output/video_analysis_results.json")
    
    # Access key outputs
    print(f"\n{'='*60}")
    print("KEY OUTPUTS FOR CONSISTENCY AUDITOR:")
    print(f"{'='*60}")
    print(f"T_actual (seconds): {results['T_actual']}")
    print(f"Severity_actual: {results['severity_actual']}")