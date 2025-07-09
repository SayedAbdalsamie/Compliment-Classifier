from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mangum import Mangum
from flask import jsonify
import torch
import os
import logging
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplimentClassifierTool:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "LLM_model")
        self.tokenizer = None
        self.model = None
        self.translator = Translator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer with error handling."""
        try:
            logger.info("Loading classifier model and tokenizer...")

            # Try to load with legacy tokenizer first
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=False,  # Use slow tokenizer to avoid compatibility issues
                    trust_remote_code=True,
                )
                logger.info("✓ Loaded tokenizer with legacy mode")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer with legacy mode: {e}")
                # Try with different parameters
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path, use_fast=False, local_files_only=True
                    )
                    logger.info("✓ Loaded tokenizer with local files only")
                except Exception as e2:
                    logger.error(f"Failed to load tokenizer: {e2}")
                    raise e2

            # Load the model
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # Use float32 for compatibility
                )
                logger.info("✓ Loaded model successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise e

            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            logger.info(f"✓ Model loaded and ready on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            self.model_loaded = False
            # Don't raise the exception, just mark as not loaded
            # This allows the service to start even if classifier fails

    def translate_to_english(self, text):
        """Translate text to English if it's Arabic."""
        if not text:
            return "No text provided"

        try:
            # if text is english don't translate
            if text.isascii():
                return text

            # if text is arabic translate to english
            detected = self.translator.detect(text)
            if detected.lang == "ar":
                translated = self.translator.translate(text, src="ar", dest="en")
                return translated.text
            return text
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text  # Return original text if translation fails

    def predict(self, text):
        """Make prediction using the loaded model."""
        if not self.model_loaded or not self.tokenizer or not self.model:
            logger.error("Model not loaded, cannot make prediction")
            return "category: Error; subcategory: Model not available; priority: Low"

        try:
            prompt = f"Complaint: {text}\nPredict:"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,  # Add max_length to prevent issues
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=1,  # Use greedy decoding for speed
                    early_stopping=True,
                )

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Prediction made successfully: {result[:50]}...")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "category: Error; subcategory: Prediction failed; priority: Low"

    def parse_prediction(self, result):
        """Parse the prediction result into structured format."""
        output = {"category": None, "subcategory": None, "priority": None}

        try:
            if not result or "Error" in result:
                output["category"] = "Error"
                output["subcategory"] = "Processing failed"
                output["priority"] = "Low"
                return output

            # Split by semicolon and parse
            parts = [p.strip() for p in result.split(";") if ":" in p]
            for part in parts:
                try:
                    key, value = part.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key == "category":
                        output["category"] = value
                    elif key == "subcategory":
                        output["subcategory"] = value
                    elif key == "priority":
                        output["priority"] = value
                except Exception as e:
                    logger.warning(f"Failed to parse part '{part}': {e}")
                    continue

            # Set defaults if parsing failed
            if not output["category"]:
                output["category"] = "General"
            if not output["subcategory"]:
                output["subcategory"] = "Other"
            if not output["priority"]:
                output["priority"] = "Medium"

        except Exception as e:
            logger.error(f"Failed to parse prediction: {e}")
            output["category"] = "Error"
            output["subcategory"] = "Parsing failed"
            output["priority"] = "Low"

        return output

    def predict_route(self, text):
        """Main prediction route that handles the complete pipeline."""
        try:
            if not self.model_loaded:
                return jsonify(
                    {
                        "category": "Error",
                        "subcategory": "Classifier not available",
                        "priority": "Low",
                        "error": "Model not loaded",
                    }
                )

            # Translate if needed
            english_text = self.translate_to_english(text)

            # Make prediction
            result = self.predict(english_text)

            # Parse result
            parsed = self.parse_prediction(result)

            logger.info(f"Classification completed: {parsed}")
            return jsonify(parsed)

        except Exception as e:
            logger.error(f"Classification route failed: {e}")
            return jsonify(
                {
                    "category": "Error",
                    "subcategory": "Processing failed",
                    "priority": "Low",
                    "error": str(e),
                }
            )


# Create a global instance for the Flask app
classifier_tool = None


def get_classifier():
    """Get or create the classifier instance."""
    global classifier_tool
    if classifier_tool is None:
        classifier_tool = ComplimentClassifierTool()
    return classifier_tool
