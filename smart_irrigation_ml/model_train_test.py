import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Extended dataset
data = [
    # Loamy/Clayey Soil
    ["Loamy/Clayey", "Paddy", "Nursery", 20, 85, 35, 22, "DAP", 20, 20, 10, "1–2", "50–100", 2, "Transplant ready at 3-leaf stage"],
    ["Loamy/Clayey", "Paddy", "Transplanting", 7, 90, 34, 24, "Urea + DAP", 40, 20, 10, "Continuous", "100–150", 3, "Flood irrigation is ideal"],
    ["Loamy/Clayey", "Paddy", "Tillering", 30, 75, 32, 20, "Urea", 30, 10, 20, "3–4", "50–80", 2, "Encourages shoot multiplication"],
    ["Loamy/Clayey", "Paddy", "Panicle Initiation", 20, 70, 30, 20, "DAP + MOP", 20, 30, 30, "3–4", "30–50", 2, "Avoid stress for yield"],
    ["Loamy/Clayey", "Paddy", "Flowering", 15, 70, 29, 20, "Foliar N Spray", 10, 10, 10, "2", "40", 2, "Needs consistent moisture"],
    ["Loamy/Clayey", "Paddy", "Grain Filling", 20, 60, 28, 18, "Potash", 5, 10, 25, "4–5", "30", 2, "Grain size development"],
    ["Loamy/Clayey", "Paddy", "Maturity", 15, 50, 27, 16, "None", 0, 0, 0, "Stop", "<20", 1, "Drain field 10 days before harvest"],
    # Alluvial Soil
    ["Alluvial", "Paddy", "Nursery", 20, 85, 35, 22, "DAP", 20, 20, 10, "1–2", "60–110", 2, "Transplant ready at 3-leaf stage"],
    ["Alluvial", "Paddy", "Transplanting", 7, 90, 34, 24, "Urea + DAP", 40, 20, 10, "Continuous", "120–160", 3, "Flood irrigation is ideal"],
    ["Alluvial", "Paddy", "Tillering", 30, 75, 32, 20, "Urea", 30, 10, 20, "3–4", "60–90", 2, "Encourages shoot multiplication"],
    ["Alluvial", "Paddy", "Panicle Initiation", 20, 70, 30, 20, "DAP + MOP", 20, 30, 30, "3–4", "40–60", 2, "Avoid stress for yield"],
    ["Alluvial", "Paddy", "Flowering", 15, 70, 29, 20, "Foliar N Spray", 10, 10, 10, "2", "50", 2, "Needs consistent moisture"],
    ["Alluvial", "Paddy", "Grain Filling", 20, 60, 28, 18, "Potash", 5, 10, 25, "4–5", "40", 2, "Grain size development"],
    ["Alluvial", "Paddy", "Maturity", 15, 50, 27, 16, "None", 0, 0, 0, "Stop", "<30", 1, "Drain field 10 days before harvest"],
]

columns = [
    "Soil Type", "Crop", "Stage", "Duration", "Ideal Moisture",
    "Max Temp", "Min Temp", "Fertilizer", "N", "P", "K",
    "Irrigation Freq", "Rainfall Tolerance", "Water Retention", "Notes"
]

df = pd.DataFrame(data, columns=columns)

class CropRequirements:
    def __init__(self, df):
        self.df = df
        self.requirements_cache = {}
        
    def get_requirements(self, soil_type, crop, stage):
        """Get ideal requirements for given soil type, crop and stage"""
        cache_key = f"{soil_type}_{crop}_{stage}"
        if cache_key in self.requirements_cache:
            return self.requirements_cache[cache_key]
            
        mask = ((self.df['Soil Type'] == soil_type) & 
                (self.df['Crop'] == crop) & 
                (self.df['Stage'] == stage))
                
        result = self.df[mask].iloc[0].to_dict() if any(mask) else None
        
        if result:
            self.requirements_cache[cache_key] = result
            return result
        else:
            raise ValueError(f"No data found for {soil_type}, {crop}, {stage}")

# Feature Engineering
def preprocess_data(df, current_moisture, current_n, current_p, current_k):
    """Add calculated features to the dataframe"""
    df = df.copy()
    
    # Calculate moisture deficit
    df['Current Moisture'] = current_moisture
    df['Moisture Deficit'] = df['Ideal Moisture'] - df['Current Moisture']
    
    # Calculate NPK deficits
    df['Current N'] = current_n
    df['Current P'] = current_p
    df['Current K'] = current_k
    df['N Deficit'] = df['N'] - df['Current N']
    df['P Deficit'] = df['P'] - df['Current P']
    df['K Deficit'] = df['K'] - df['Current K']
    
    # Create irrigation need label (1 if moisture deficit > 15 or water retention < 2)
    df['Irrigation Needed'] = ((df['Moisture Deficit'] > 15) | (df['Water Retention'] < 2)).astype(int)
    
    # Create fertilizer need labels
    df['N Needed'] = (df['N Deficit'] > 5).astype(int)
    df['P Needed'] = (df['P Deficit'] > 3).astype(int)
    df['K Needed'] = (df['K Deficit'] > 2).astype(int)
    
    return df

# Initialize crop requirements
crop_reqs = CropRequirements(df)

# Define features and targets
categorical_features = ["Soil Type", "Crop", "Stage"]
numeric_features = [
    "Duration", "Ideal Moisture", "Max Temp", "Min Temp", 
    "Water Retention", "Current Moisture", "Current N", 
    "Current P", "Current K", "Moisture Deficit",
    "N Deficit", "P Deficit", "K Deficit"
]

# Recommendation System
class IrrigationAdvisor:
    def __init__(self, crop_reqs):
        self.crop_reqs = crop_reqs
        
        # Train models when initialized
        self._train_models()
    
    def _train_models(self):
        """Train all required models"""
        # We'll simulate training data based on various current conditions
        # In a real system, you'd have actual field measurements
        simulated_data = []
        
        # Generate simulated data for different conditions
        for _, row in self.crop_reqs.df.iterrows():
            for current_moisture in np.linspace(20, 100, 5):
                for current_n in np.linspace(0, row['N']*1.5, 3):
                    for current_p in np.linspace(0, row['P']*1.5, 3):
                        for current_k in np.linspace(0, row['K']*1.5, 3):
                            simulated_data.append({
                                **row.to_dict(),
                                'Current Moisture': current_moisture,
                                'Current N': current_n,
                                'Current P': current_p,
                                'Current K': current_k
                            })
        
        sim_df = pd.DataFrame(simulated_data)
        sim_df = preprocess_data(sim_df, 
                                sim_df['Current Moisture'], 
                                sim_df['Current N'], 
                                sim_df['Current P'], 
                                sim_df['Current K'])
        
        X = sim_df[categorical_features + numeric_features]
        y_irrigation = sim_df['Irrigation Needed']
        y_fertilizer = sim_df[['N Needed', 'P Needed', 'K Needed']]
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numeric_features)
            ])
        
        # Create and train models
        self.irrigation_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]).fit(X, y_irrigation)
        
        self.fertilizer_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]).fit(X, y_fertilizer)
        
        # Irrigation amount model
        y_amount = sim_df['Moisture Deficit'] * sim_df['Water Retention']
        self.irrigation_amount_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ]).fit(X, y_amount)
    
    def get_recommendations(self, field_data, weather_forecast):
        """
        field_data: dict containing basic field conditions
        weather_forecast: list of dicts containing hourly/daily forecast
        """
        # Get ideal requirements
        requirements = self.crop_reqs.get_requirements(
            field_data["Soil Type"],
            field_data["Crop"],
            field_data["Stage"]
        )
        
        # Prepare complete input data
        complete_data = {
            **requirements,
            "Current Moisture": field_data["Current Moisture"],
            "Current N": field_data["Current N"],
            "Current P": field_data["Current P"],
            "Current K": field_data["Current K"]
        }
        
        # Calculate derived features
        input_df = preprocess_data(pd.DataFrame([complete_data]), 
                                 complete_data["Current Moisture"],
                                 complete_data["Current N"],
                                 complete_data["Current P"],
                                 complete_data["Current K"])
        
        # Check for upcoming rain
        upcoming_rain = self._check_upcoming_rain(weather_forecast)
        
        # Make predictions
        irrigation_needed = self.irrigation_model.predict(input_df)[0]
        fertilizer_needs = self.fertilizer_model.predict(input_df)[0]
        irrigation_amount = self.irrigation_amount_model.predict(input_df)[0]
        
        # Generate recommendations
        recommendations = {
            "irrigation": {
                "needed": bool(irrigation_needed),
                "recommended_amount": max(0, irrigation_amount),
                "consider_rain": upcoming_rain,
                "final_recommendation": self._get_irrigation_final_recommendation(
                    irrigation_needed, irrigation_amount, upcoming_rain),
                "ideal_moisture": requirements["Ideal Moisture"]
            },
            "fertilizer": {
                "N": bool(fertilizer_needs[0]),
                "P": bool(fertilizer_needs[1]),
                "K": bool(fertilizer_needs[2]),
                "recommendations": self._get_fertilizer_recommendations(
                    fertilizer_needs, 
                    input_df.iloc[0]),
                "ideal_N": requirements["N"],
                "ideal_P": requirements["P"],
                "ideal_K": requirements["K"]
            }
        }
        
        return recommendations
    
    def _check_upcoming_rain(self, weather_forecast):
        """Check if significant rain is expected in the next 24 hours"""
        if not weather_forecast:
            return False
        
        # Sum expected rainfall in next 24 hours
        total_rain = sum(float(hour.get("Rain Amount (mm)", 0)) for hour in weather_forecast[:24])
        return total_rain > 5  # Consider significant if >5mm
    
    def _get_irrigation_final_recommendation(self, needed, amount, upcoming_rain):
        if not needed:
            return "No irrigation needed at this time."
        
        if upcoming_rain:
            return f"Delay irrigation. Significant rain expected soon. Would normally recommend {amount:.1f}mm."
        
        return f"Irrigate with {amount:.1f}mm of water."
    
    def _get_fertilizer_recommendations(self, needs, input_data):
        recommendations = []
        nutrients = ['N', 'P', 'K']
        
        for i, nutrient in enumerate(nutrients):
            if needs[i]:
                deficit = input_data.get(f"{nutrient} Deficit", 0)
                recommendations.append(
                    f"Apply {max(0, deficit):.1f} kg/ha of {nutrient}-based fertilizer."
                )
        
        return recommendations if recommendations else ["No fertilizer needed at this time."]

# Example Usage
advisor = IrrigationAdvisor(crop_reqs)

# Field data input - now only needs current conditions and identifiers
field_data = {
    "Soil Type": "Alluvial",
    "Crop": "Paddy",
    "Stage": "Panicle Initiation",
    "Current Moisture": 40,  # Actual field measurement
    "Current N": 10,  # Soil test results
    "Current P": 12,
    "Current K": 7
}

# Weather forecast data (simplified)
weather_forecast = [
    {"Date Time": "2023-06-01 12:00", "Temperature (°C)": 28, "Rain Amount (mm)": 0},
    {"Date Time": "2023-06-01 15:00", "Temperature (°C)": 30, "Rain Amount (mm)": 0},
    {"Date Time": "2023-06-01 18:00", "Temperature (°C)": 25, "Rain Amount (mm)": 5},  # Rain coming
    {"Date Time": "2023-06-01 21:00", "Temperature (°C)": 22, "Rain Amount (mm)": 10},
]

# Get recommendations
recommendations = advisor.get_recommendations(field_data, weather_forecast)

# Print results
print("\n=== Irrigation Recommendation ===")
print(f"Current moisture: {field_data['Current Moisture']}")
print(f"Ideal moisture: {recommendations['irrigation']['ideal_moisture']}")
print(f"Irrigation needed: {'Yes' if recommendations['irrigation']['needed'] else 'No'}")
print(f"Recommended amount: {recommendations['irrigation']['recommended_amount']:.1f}mm")
print(f"Upcoming rain: {'Yes' if recommendations['irrigation']['consider_rain'] else 'No'}")
print(f"Final advice: {recommendations['irrigation']['final_recommendation']}")

print("\n=== Fertilizer Recommendation ===")
print(f"Current N: {field_data['Current N']} (Ideal: {recommendations['fertilizer']['ideal_N']})")
print(f"Current P: {field_data['Current P']} (Ideal: {recommendations['fertilizer']['ideal_P']})")
print(f"Current K: {field_data['Current K']} (Ideal: {recommendations['fertilizer']['ideal_K']})")
for rec in recommendations['fertilizer']['recommendations']:
    print("-", rec)