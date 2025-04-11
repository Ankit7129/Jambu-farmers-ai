from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from smart_irrigation_ml.components.advisor import get_recommendations

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

app = FastAPI(
    title="Jambavantha Smart Irrigation API",
    description="Gives smart irrigation and fertilizer recommendations",
    version="1.0.0"
)

# ✅ Add this block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://jambagrad.com"],  # 👈 Allow your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/get-recommendations")
async def recommend_irrigation(input_data: InputData):
    try:
        field_data = {
            "Soil Type": input_data.field_data.Soil_Type,
            "Crop": input_data.field_data.Crop,
            "Stage": input_data.field_data.Stage,
            "Current Moisture": input_data.field_data.Current_Moisture,
            "Current N": input_data.field_data.Current_N,
            "Current P": input_data.field_data.Current_P,
            "Current K": input_data.field_data.Current_K,
        }

        weather_forecast = [
            {
                "Date Time": w.Date_Time,
                "Temperature (°C)": w.Temperature_C,
                "Rain Amount (mm)": w.Rain_Amount_mm
            } for w in input_data.weather_forecast
        ]

        recommendations = get_recommendations(field_data, weather_forecast)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to Jambavantha Smart Irrigation API!"}
