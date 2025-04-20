from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load Gemini API Key
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API Key not found in environment!")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Pydantic models
class FieldData(BaseModel):
    Soil_Type: str
    Crop: str
    Stage: str
    Current_Moisture: float
    Current_N: float
    Current_P: float
    Current_K: float

class WeatherData(BaseModel):
    Date_Time: str
    Temperature_C: float
    Rain_Amount_mm: float

class InputData(BaseModel):
    field_data: FieldData
    weather_forecast: List[WeatherData]
    language: Optional[str] = "hindi"

# Initialize FastAPI app
app = FastAPI(
    title="Jambavantha Smart Irrigation API",
    description="Gives smart irrigation and fertilizer recommendations",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Advisor class
class GeminiAdvisor:
    def __init__(self):
        self.base_prompt = """
You are an agricultural expert system providing irrigation and fertilizer recommendations. 
Always respond in JSON format exactly matching this structure:

{
    "metadata": {
        "original_request": {"soil_type": "", "crop": "", "stage": ""},
        "actual_match": {"soil_type": "", "crop": "", "stage": ""},
        "is_approximate": bool,
        "notes": "",
        "success": true
    },
    "recommendatio   ns": {
        "irrigation": {
            "current": float,
            "ideal": float,
            "status": "Adequate/Slightly low/Moderately low/Critically low",
            "needed": bool,
            "amount": float,
            "upcoming_rain": bool,
            "advice": "",
            "details": {
                "water_retention": int,
                "rainfall_tolerance": "",
                "irrigation_frequency": ""
            }
        },
        "fertilizer": {
            "N": {
                "needed": bool,
                "current": float,
                "ideal": float,
                "status": "Sufficient/Slightly deficient/Deficient",
                "recommendation": "",
                "fertilizer_type": "",
                "deficit": float
            },
            "P": {
                "needed": bool,
                "current": float,
                "ideal": float,
                "status": "Sufficient/Slightly deficient/Deficient",
                "recommendation": "",
                "fertilizer_type": "",
                "deficit": float
            },
            "K": {
                "needed": bool,
                "current": float,
                "ideal": float,
                "status": "Sufficient/Slightly deficient/Deficient",
                "recommendation": "",
                "fertilizer_type": "",
                "deficit": float
            },
            "general_advice": ""
        }
    }
}

Important rules:
1. All numerical values must be realistic for agriculture
2. Status levels must follow the exact wording above
3. For language {language}, translate all text fields (status, advice, etc.)
4. Never change the JSON structure, only translate text values
"""

    def get_recommendations(self, input_data: Dict) -> Dict:
        try:
            # Fill in the language into the prompt safely
            prompt = f"""{self.base_prompt.replace('{language}', input_data.get('language', 'english'))}

Current Conditions:
- Soil: {input_data['field_data']['Soil_Type']}
- Crop: {input_data['field_data']['Crop']} ({input_data['field_data']['Stage']} stage)
- Moisture: {input_data['field_data']['Current_Moisture']}%
- Nutrients (N/P/K): {input_data['field_data']['Current_N']}/{input_data['field_data']['Current_P']}/{input_data['field_data']['Current_K']}

Weather Forecast:
{self._format_weather(input_data['weather_forecast'])}

Provide recommendations in {input_data.get('language', 'english')}.
"""

            # Send request to Gemini
            response = requests.post(
                GEMINI_URL,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()
            candidates = result.get("candidates", [])
            if not candidates:
                raise ValueError("No response from Gemini API")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("Invalid response format from Gemini")

            text_response = parts[0].get("text", "")

            try:
                return json.loads(text_response)
            except:
                return self._extract_json_from_text(text_response)

        except Exception as e:
            return {
                "metadata": {
                    "success": False,
                    "error": str(e),
                    "original_request": input_data.get('field_data', {})
                }
            }

    def _format_weather(self, forecast: List[Dict]) -> str:
        if not forecast:
            return "No weather forecast available"
        return "\n".join(
            f"- {w['Date_Time']}: {w['Temperature_C']}°C, Rain: {w['Rain_Amount_mm']}mm"
            for w in forecast[:24]
        )

    def _extract_json_from_text(self, text: str) -> Dict:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except:
            return {
                "metadata": {
                    "success": False,
                    "error": "Could not parse Gemini JSON response",
                    "original_request": {}
                }
            }

# Singleton advisor instance
advisor = GeminiAdvisor()

# Root route
@app.get("/")
def root():
    return {"message": "Welcome to Jambavantha Smart Irrigation API!"}

# POST endpoint
@app.post("/get-recommendations")
def get_advice(data: InputData):
    result = advisor.get_recommendations(data.dict())
    return result