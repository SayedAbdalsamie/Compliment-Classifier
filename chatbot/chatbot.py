from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import logging
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class ChatBotTool:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("GEMINI_API_KEY is not set in the environment variables")
            raise ValueError("GEMINI_API_KEY is not set in the environment variables")

        genai.configure(api_key=self.api_key)
        self.model = None
        self.model_name = None
        self.load_model()

    def load_model(self):
        """Load the Gemini model with fallback options."""
        models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

        for model_name in models_to_try:
            try:
                self.model = genai.GenerativeModel(model_name)
                self.model_name = model_name
                logger.info(f"Successfully loaded {model_name} model")
                return self.model
            except Exception as e:
                logger.warning(f"Error loading {model_name} model: {e}")
                continue

        logger.error("Failed to load any Gemini model")
        self.model = None
        return None

    def get_icsm_context(self):
        """Get the ICMS context information."""
        return """
        You are an AI assistant providing information about the Integrated City Management System (ICMS).

        PROJECT OVERVIEW:
        The Integrated City Management System (ICMS) is a unified platform designed to enhance communication, service delivery, and engagement between city residents and local authorities. The system simplifies government interactions, improves city infrastructure management, and fosters community interaction through a range of digital services.

        CORE OBJECTIVES:
        - Simplify interaction between residents and government services
        - Centralize government tasks, city navigation, and infrastructure maintenance
        - Promote smart, efficient, and transparent city management
        - Enhance community engagement and social interaction

        TARGET AUDIENCE:
        - City residents seeking government services
        - Government officials managing city operations
        - Service providers delivering public services

        KEY SUBSYSTEMS:

        1. GOVERNMENT TASKS WEBSITE:
        - Centralized web portal for citizen services
        - Services: tax payments, permits, appointments, document verification
        - Features: citizen ID-based login, integrated payment system, document download
        - Impact: Streamlined government processes for better efficiency

        2. CITY CHAT APP:
        - Real-time issue reporting and tracking
        - Services: electrical, plumbing, infrastructure issues
        - Features: AI bot categorization, task assignment, problem resolution suggestions
        - Additional: Booking services (garbage collection, park maintenance)
        - Impact: Faster response and issue resolution

        3. CITIZEN SOCIAL MEDIA PLATFORM:
        - Community engagement and local discussions
        - Features: user-generated content, event management, public/private posts
        - Impact: Secure communication and community engagement

        4. PUBLIC SERVICES WEBSITE:
        - Service booking and management
        - Services: garbage collection, park maintenance
        - Features: automatic notifications, service feedback, real-time tracking
        - Impact: Improved service delivery and citizen satisfaction

        BENEFITS:
        - For Citizens: Simplified city management, easy access to services, better communication
        - For City Officials: Efficient tracking, real-time updates, data-driven decision-making
        - For Community: Enhanced engagement through social media, better local interaction

        TECHNICAL FEATURES:
        - AI-powered problem categorization and resolution
        - Real-time tracking and notifications
        - Integrated payment systems
        - Mobile-responsive design
        - Multi-language support
        - Secure authentication and data protection

        RESPONSE GUIDELINES:
        - Provide direct, helpful answers about ICMS
        - Use natural, conversational language
        - Focus on practical benefits and use cases
        - Include specific examples when relevant
        - Avoid generic responses - be specific to ICMS
        """

    def ask(self, question):
        """Ask a question about ICMS and return a response."""
        if not question or not question.strip():
            return {"error": "No question provided"}

        question = question.strip()
        logger.info(f"Processing question: {question[:50]}...")

        # Check if model is loaded
        if self.model is None:
            return {
                "error": "Gemini model not available. Please check your API key and model configuration.",
                "model_status": "not_loaded",
                "api_key_set": bool(self.api_key),
            }

        try:
            # Get context
            icsm_context = self.get_icsm_context()

            # Try multiple approaches for generating response
            response_text = None
            last_error = None

            # Approach 1: Simple direct question (most reliable)
            try:
                logger.info("Trying simple direct question approach...")
                simple_prompt = f"Answer this question about ICMS (Integrated City Management System): {question}"
                response = self.model.generate_content(simple_prompt)
                response_text = response.text
                logger.info("Response generated using simple prompt")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Simple prompt failed: {e}")

            # Approach 2: System instruction (if simple failed)
            if not response_text:
                try:
                    logger.info("Trying system instruction approach...")
                    response = self.model.generate_content(
                        prompt=question,
                        system_instruction=icsm_context,
                        generation_config={
                            "temperature": 0.7,
                            "top_p": 0.8,
                            "top_k": 40,
                            "max_output_tokens": 1024,
                        },
                    )
                    response_text = response.text
                    logger.info("Response generated using system instruction")
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"System instruction failed: {e}")

            # Approach 3: Prompt method (last resort)
            if not response_text:
                try:
                    logger.info("Trying prompt method approach...")
                    prompt = f"{icsm_context}\n\nUser's Question: {question}\n\nProvide a direct and helpful answer about ICMS. Use natural, conversational language and focus on practical benefits."

                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.7,
                            "top_p": 0.8,
                            "top_k": 40,
                            "max_output_tokens": 1024,
                        },
                    )
                    response_text = response.text
                    logger.info("Response generated using prompt method")
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"Prompt method also failed: {e}")

            if response_text:
                return {
                    "answer": response_text,
                    "model_used": self.model_name,
                    "status": "success",
                }
            else:
                return {
                    "error": f"Failed to generate response. Last error: {last_error}",
                    "model_status": "error",
                    "api_key_set": bool(self.api_key),
                    "model_name": self.model_name,
                }

        except Exception as e:
            logger.error(f"Error in ask method: {e}")
            return {
                "error": f"Failed to generate response: {str(e)}",
                "model_status": "error",
                "api_key_set": bool(self.api_key),
                "model_name": self.model_name,
            }

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "api_key_set": bool(self.api_key),
        }

    def test_connection(self):
        """Test the connection to Gemini API."""
        try:
            if not self.model:
                return {"status": "error", "message": "Model not loaded"}

            # Simple test question
            test_response = self.ask("What is ICMS?")
            if "error" in test_response:
                return {"status": "error", "message": test_response["error"]}
            else:
                return {"status": "success", "message": "Connection working"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Create a global instance for the Flask app
chatbot_tool = None


def get_chatbot():
    """Get or create the chatbot instance."""
    global chatbot_tool
    if chatbot_tool is None:
        try:
            chatbot_tool = ChatBotTool()
        except Exception as e:
            logger.error(f"Failed to create chatbot instance: {e}")
            chatbot_tool = None
    return chatbot_tool
