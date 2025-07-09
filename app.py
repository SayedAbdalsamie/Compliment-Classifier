"""
Unified Flask API for ICMS Services
Combines Chatbot, Compliment Classifier, and Text Moderation services
"""

from flask import Flask, request, jsonify, Response
import json
from flask_cors import CORS
import logging
import traceback
import os
import sys
from googletrans import Translator

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import the tool modules
try:
    from chatbot.chatbot import get_chatbot
    from classifer.classifer import get_classifier
    from moderator.moderator import (
        initialize_moderator,
        moderate_single_text,
        moderate_batch_texts,
        get_models_info,
        get_system_status,
    )

    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    get_chatbot = None
    get_classifier = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add a translation utility
translator = Translator()


def translate_arabic_to_english(text):
    if not text:
        return text
    try:
        # If text is ASCII, assume it's not Arabic
        if text.isascii():
            return text
        detected = translator.detect(text)
        if detected.lang == "ar":
            translated = translator.translate(text, src="ar", dest="en")
            return translated.text
        return text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


# Helper to return JSON with ensure_ascii=False


def jsonify_arabic(data, status=200):
    return Response(
        json.dumps(data, ensure_ascii=False), status=status, mimetype="application/json"
    )


# Global service instances
chatbot = None
classifier = None
moderator_initialized = False


def initialize_services():
    """Initialize all services."""
    global chatbot, classifier, moderator_initialized

    success_count = 0

    # Initialize Chatbot
    try:
        logger.info("Initializing Chatbot service...")
        chatbot = get_chatbot()
        if chatbot:
            logger.info("✓ Chatbot service initialized successfully")
            success_count += 1
        else:
            logger.error("✗ Failed to initialize Chatbot service")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Chatbot: {e}")
        chatbot = None

    # Initialize Classifier
    try:
        logger.info("Initializing Compliment Classifier service...")
        classifier = get_classifier()
        logger.info("✓ Compliment Classifier service initialized successfully")
        success_count += 1
    except Exception as e:
        logger.error(f"✗ Failed to initialize Classifier: {e}")
        classifier = None

    # Initialize Moderator
    try:
        logger.info("Initializing Text Moderation service...")
        moderator_initialized = initialize_moderator()
        if moderator_initialized:
            logger.info("✓ Text Moderation service initialized successfully")
            success_count += 1
        else:
            logger.error("✗ Failed to initialize Text Moderation service")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Text Moderation: {e}")
        moderator_initialized = False

    logger.info(f"Services initialized: {success_count}/3")
    return success_count


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint for all services."""
    status = {
        "status": "online",
        "services": {
            "chatbot": "online" if chatbot else "offline",
            "classifier": "online" if classifier else "offline",
            "moderator": "online" if moderator_initialized else "offline",
        },
        "total_services": 3,
        "active_services": sum(
            [
                1 if chatbot else 0,
                1 if classifier else 0,
                1 if moderator_initialized else 0,
            ]
        ),
    }
    return jsonify(status)


@app.route("/chatbot", methods=["POST"])
def chatbot_route():
    """
    Chatbot service endpoint.

    Expected JSON payload:
    {
        "question": "Your question about ICMS"
    }

    Returns:
    {
        "answer": "AI response about ICMS"
    }
    """
    try:
        # Check if chatbot is initialized
        if not chatbot:
            return jsonify({"error": "Chatbot service not available"}), 503

        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract question
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Translate if Arabic
        question = translate_arabic_to_english(question)
        # Always translate answer to Arabic
        answer = ""
        try:
            answer = translator.translate(question, src="en", dest="ar").text
        except Exception as e:
            logger.warning(f"Failed to translate question to Arabic: {e}")

        # Get response from chatbot
        response = chatbot.ask(question)

        # Handle response - the chatbot now returns a dict
        if isinstance(response, dict):
            if "error" in response:
                # Return error response with more details
                return jsonify_arabic(response, 500)
            elif "answer" in response:
                # Always translate answer to Arabic
                answer = response["answer"]
                try:
                    answer = translator.translate(answer, src="en", dest="ar").text
                except Exception as e:
                    logger.warning(f"Failed to translate answer to Arabic: {e}")
                return jsonify_arabic(
                    {
                        "answer": answer,
                        "model_used": response.get("model_used", "Unknown"),
                        "status": response.get("status", "success"),
                    }
                )
            else:
                # Unexpected response format
                return (
                    jsonify({"error": "Unexpected response format from chatbot"}),
                    500,
                )
        else:
            # Fallback for other response types
            answer = str(response)
            try:
                answer = translator.translate(answer, src="en", dest="ar").text
            except Exception as e:
                logger.warning(f"Failed to translate answer to Arabic: {e}")
            return jsonify_arabic({"answer": answer})

    except Exception as e:
        logger.error(f"Error in chatbot route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Chatbot processing failed: {str(e)}"}), 500


@app.route("/chatbot/debug", methods=["GET"])
def chatbot_debug():
    """Debug endpoint to check chatbot status."""
    try:
        if not chatbot:
            return jsonify({"status": "error", "message": "Chatbot not initialized"})

        # Get model info
        model_info = chatbot.get_model_info()

        return jsonify(
            {
                "status": "success",
                "model_info": model_info,
                "api_key_set": bool(
                    chatbot.api_key if hasattr(chatbot, "api_key") else False
                ),
            }
        )

    except Exception as e:
        logger.error(f"Error in chatbot debug route: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route("/classifier", methods=["POST"])
def classifier_route():
    """
    Compliment Classifier service endpoint.

    Expected JSON payload:
    {
        "text": "Your complaint text"
    }

    Returns:
    {
        "category": "Category name",
        "subcategory": "Subcategory name",
        "priority": "Priority level"
    }
    """
    try:
        # Check if classifier is initialized
        if not classifier:
            return jsonify({"error": "Classifier service not available"}), 503

        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract text
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Translate if Arabic
        text = translate_arabic_to_english(text)

        # Get prediction from classifier
        result = classifier.predict_route(text)

        # Handle response
        if hasattr(result, "json"):
            return result
        else:
            return jsonify(result)

    except Exception as e:
        logger.error(f"Error in classifier route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Classifier processing failed"}), 500


@app.route("/moderator", methods=["POST"])
def moderator_route():
    """
    Text Moderation service endpoint.

    Expected JSON payload:
    {
        "text": "Text to moderate"
    }

    Returns:
    {
        "status": "Accept" or "Reject",
        "confidence": 0.85
    }
    """
    try:
        # Check if moderator is initialized
        if not moderator_initialized:
            return jsonify({"error": "Moderator service not available"}), 503

        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract text
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Translate if Arabic
        text = translate_arabic_to_english(text)

        # Moderate the text
        result = moderate_single_text(text)

        # Return response
        if result.get("error"):
            return jsonify({"error": result["error"]}), 500

        return jsonify({"status": result["status"], "confidence": result["confidence"]})

    except Exception as e:
        logger.error(f"Error in moderator route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Moderation processing failed"}), 500


@app.route("/moderator/batch", methods=["POST"])
def moderator_batch_route():
    """
    Batch Text Moderation service endpoint.

    Expected JSON payload:
    {
        "texts": ["Text 1", "Text 2", ...]
    }

    Returns:
    {
        "success": true,
        "results": [...]
    }
    """
    try:
        # Check if moderator is initialized
        if not moderator_initialized:
            return jsonify({"error": "Moderator service not available"}), 503

        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract texts
        texts = data.get("texts", [])
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "No texts provided or texts is not a list"}), 400

        # Translate each if Arabic
        texts = [translate_arabic_to_english(t) for t in texts]

        # Limit batch size
        if len(texts) > 100:
            return (
                jsonify({"error": "Batch size too large. Maximum 100 texts allowed."}),
                400,
            )

        # Moderate all texts
        result = moderate_batch_texts(texts)

        if not result.get("success"):
            return (
                jsonify({"error": result.get("error", "Batch processing failed")}),
                500,
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in moderator batch route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Batch moderation processing failed"}), 500


@app.route("/moderator/models", methods=["GET"])
def moderator_models_route():
    """Get information about loaded moderation models."""
    try:
        if not moderator_initialized:
            return jsonify({"error": "Moderator service not available"}), 503

        result = get_models_info()

        if not result.get("success"):
            return (
                jsonify({"error": result.get("error", "Failed to get models info")}),
                500,
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in moderator models route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to get models information"}), 500


@app.route("/status", methods=["GET"])
def detailed_status():
    """Get detailed status of all services."""
    try:
        status = {"overall_status": "online", "services": {}}

        # Chatbot status
        if chatbot:
            status["services"]["chatbot"] = {"status": "online", "model": "Gemini AI"}
        else:
            status["services"]["chatbot"] = {
                "status": "offline",
                "error": "Service not initialized",
            }

        # Classifier status
        if classifier:
            status["services"]["classifier"] = {
                "status": "online",
                "model": "Custom LLM",
            }
        else:
            status["services"]["classifier"] = {
                "status": "offline",
                "error": "Service not initialized",
            }

        # Moderator status
        if moderator_initialized:
            moderator_status = get_system_status()
            status["services"]["moderator"] = {
                "status": "online",
                "models_count": moderator_status.get("models", 0),
                "model_names": moderator_status.get("model_names", []),
            }
        else:
            status["services"]["moderator"] = {
                "status": "offline",
                "error": "Service not initialized",
            }

        return jsonify(status)

    except Exception as e:
        logger.error(f"Error in detailed status route: {e}")
        return jsonify({"error": "Failed to get detailed status"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Warn if GEMINI_API_KEY is not set
    if not os.getenv("GEMINI_API_KEY"):
        print(
            "⚠️  Warning: GEMINI_API_KEY is not set. The chatbot service will not work."
        )
    # Initialize all services
    print("Initializing ICMS Unified Services...")
    services_initialized = initialize_services()

    if services_initialized == 0:
        print("❌ No services initialized successfully!")
        print("Please check your configuration and dependencies.")
        exit(1)

    # print(f"✅ {services_initialized}/3 services initialized successfully!")
    # print()
    # print("🚀 Starting ICMS Unified API Server...")
    # print("Available endpoints:")
    # print("  GET  /                    - Health check")
    # print("  GET  /status              - Detailed service status")
    # print("  POST /chatbot             - Chatbot service")
    # print("  POST /classifier          - Compliment classifier service")
    # print("  POST /moderator           - Text moderation service")
    # print("  POST /moderator/batch     - Batch text moderation")
    # print("  GET  /moderator/models    - Moderation models info")
    # print()
    # print("📝 Example usage:")
    # print("  curl -X POST http://localhost:5000/chatbot \")
    # print("       -H 'Content-Type: application/json' \")
    # print('       -d '{"question": "What is ICMS?"}')
    # print()

    # Run the Flask app
    app.run(
        host="0.0.0.0",  # Allow external connections
        port=5000,  # Default Flask port
        # debug=True,  # Enable debug mode
    )
