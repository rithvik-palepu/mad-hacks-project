import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NLPProcessor")

# --- OCR STRATEGY PATTERN ---

class OCREngine(ABC):
    """Abstract base class to allow swapping OCR engines (Print vs Handwriting)."""
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        pass

class TesseractEngine(OCREngine):
    """
    Standard OCR for PRINTED text.
    Pros: Fast, lightweight, runs on CPU.
    Cons: Fails significantly on cursive/handwriting.
    """
    def __init__(self, cmd_path: Optional[str] = None):
        self.enabled = False
        try:
            import pytesseract
            from PIL import Image
            if cmd_path:
                pytesseract.pytesseract.tesseract_cmd = cmd_path
            self.pytesseract = pytesseract
            self.Image = Image
            self.enabled = True
        except ImportError:
            logger.error("Tesseract/PIL not installed. OCR disabled.")

    def extract_text(self, image_path: str) -> str:
        if not self.enabled:
            return ""
        try:
            img = self.Image.open(image_path)
            # psm 6 assumes a single uniform block of text
            return self.pytesseract.image_to_string(img, config='--psm 6')
        except Exception as e:
            logger.error(f"Tesseract Extraction Failed: {e}")
            return ""

class HandwritingEngine(OCREngine):
    """
    Advanced OCR for HANDWRITTEN text (Cursive/Print).
    Uses Microsoft's TrOCR (Transformer-based Optical Character Recognition).
    
    NOTE: This requires 'transformers' and 'torch' libraries.
    """
    def __init__(self):
        self.processor = None
        self.model = None
        self.enabled = False
        
        # In a real deployment, you would uncomment the imports below.
        # We keep them commented to ensure the demo runs without heavy dependencies.
        """
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image
            import torch
            
            logger.info("Loading TrOCR Model (microsoft/trocr-base-handwritten)...")
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            self.Image = Image
            self.torch = torch
            self.enabled = True
        except ImportError:
            logger.warning("Transformers/Torch not found. Handwriting OCR disabled.")
        """
        if not self.enabled:
            logger.warning("Handwriting Engine is currently a placeholder. Install 'transformers' to enable.")

    def extract_text(self, image_path: str) -> str:
        if not self.enabled:
            return "[HANDWRITING OCR NOT LOADED]"
        
        # TrOCR Logic (Skeleton)
        try:
            image = self.Image.open(image_path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
        except Exception as e:
            logger.error(f"TrOCR Failed: {e}")
            return ""

# --- MAIN PROCESSOR ---

class ReportProcessor:
    def __init__(self, use_handwriting_model: bool = False, tesseract_cmd: Optional[str] = None):
        """
        Args:
            use_handwriting_model: If True, attempts to load TrOCR. If False, uses Tesseract.
        """
        self.severity_keywords = {
            "severe": ["severe", "fatal", "critical", "major", "crushed", "destroyed"],
            "moderate": ["moderate", "medium", "dent", "bumper"],
            "minor": ["minor", "scratch", "fender bender", "light", "scuff"]
        }
        self.type_keywords = ["head-on", "rear-end", "sideswipe", "collision", "T-bone"]

        # Select Engine Strategy
        if use_handwriting_model:
            logger.info("Initializing Advanced Handwriting Engine...")
            self.ocr_engine = HandwritingEngine()
        else:
            logger.info("Initializing Standard Tesseract Engine...")
            self.ocr_engine = TesseractEngine(cmd_path=tesseract_cmd)

    def _parse_time_to_seconds(self, time_str: str) -> int:
        """Parses time strings (12h/24h) into total seconds."""
        time_str = time_str.upper().strip()
        # Handle common OCR errors (e.g., 'O' instead of '0')
        time_str = time_str.replace("O", "0").replace("o", "0")
        
        is_pm = "PM" in time_str
        is_am = "AM" in time_str
        
        clean_time = re.sub(r"[^0-9:]", "", time_str)
        parts = clean_time.split(":")
        
        try:
            if len(parts) < 2: return -1
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2]) if len(parts) > 2 else 0
            
            if (is_pm and hours < 12): hours += 12
            if (is_am and hours == 12): hours = 0
                
            return (hours * 3600) + (minutes * 60) + seconds
        except ValueError:
            return -1

    def extract_time(self, text: str) -> int:
        # 1. Look for labeled time fields often found in forms
        form_regex = r"(?:Time|at)[:\s\.]*(\d{1,2}\s*[:\.]\s*\d{2})(?:\s*[:\.]\s*\d{2})?\s*(AM|PM)?"
        match = re.search(form_regex, text, re.IGNORECASE)
        
        if match:
            time_str = match.group(1).replace(".", ":") # Fix common OCR error 10.30 -> 10:30
            if match.group(2):
                time_str += f" {match.group(2)}"
            logger.info(f"Time found (Contextual): {time_str}")
            return self._parse_time_to_seconds(time_str)

        # 2. Look for standalone time patterns
        general_regex = r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b"
        match = re.search(general_regex, text)
        if match:
            logger.info(f"Time found (General): {match.group(1)}")
            return self._parse_time_to_seconds(match.group(1))

        return -1

    def extract_severity(self, text: str) -> str:
        text_lower = text.lower()
        # Normalize common handwriting OCR mistakes
        # e.g., "sever" -> "severe", "nimer" -> "minor"
        text_lower = text_lower.replace("sever ", "severe ").replace("mincr", "minor")
        
        for sev_level, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    logger.info(f"Severity found: {sev_level.title()} (match: {keyword})")
                    return sev_level.title()
        return "Unknown"

    def process_report(self, input_data: str, is_image_path: bool = False) -> Dict[str, Any]:
        if is_image_path:
            raw_text = self.ocr_engine.extract_text(input_data)
        else:
            raw_text = input_data

        if not raw_text:
            return {"error": "No text to process"}

        # Clean up raw text (remove newlines inside sentences, etc)
        clean_text = raw_text.replace("\n", " ")

        return {
            "TReport": self.extract_time(clean_text),
            "SeverityReport": self.extract_severity(clean_text),
            "RawTextSnippet": clean_text[:100] + "..."
        }
    