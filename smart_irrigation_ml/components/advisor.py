import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .crop_dataset import get_crop_data
from .crop_requirements import CropRequirements


def preprocess_data(df, current_moisture, current_n, current_p, current_k):
    """Preprocess input data with current field conditions"""
    df = df.copy()
    df['Current Moisture'] = current_moisture
    df['Moisture Deficit'] = df['Ideal Moisture'] - df['Current Moisture']
    
    df['Current N'] = current_n
    df['Current P'] = current_p
    df['Current K'] = current_k
    df['N Deficit'] = df['N'] - df['Current N']
    df['P Deficit'] = df['P'] - df['Current P']
    df['K Deficit'] = df['K'] - df['Current K']
    
    df['Irrigation Needed'] = ((df['Moisture Deficit'] > 15) | (df['Water Retention'] < 2)).astype(int)
    df['N Needed'] = (df['N Deficit'] > 5).astype(int)
    df['P Needed'] = (df['P Deficit'] > 3).astype(int)
    df['K Needed'] = (df['K Deficit'] > 2).astype(int)
    
    return df


class IrrigationAdvisor:
    def __init__(self, crop_reqs):
        """Initialize the advisor with crop requirements and trained models"""
        self.crop_reqs = crop_reqs
        self._train_models()

    def _train_models(self):
        """Train machine learning models for recommendations"""
        # Generate simulated training data
        simulated_data = []
        for _, row in self.crop_reqs.df.iterrows():
            for current_moisture in np.linspace(20, 100, 5):
                for current_n in np.linspace(0, row['N'] * 1.5, 3):
                    for current_p in np.linspace(0, row['P'] * 1.5, 3):
                        for current_k in np.linspace(0, row['K'] * 1.5, 3):
                            simulated_data.append({
                                **row.to_dict(),
                                'Current Moisture': current_moisture,
                                'Current N': current_n,
                                'Current P': current_p,
                                'Current K': current_k
                            })

        sim_df = pd.DataFrame(simulated_data)
        sim_df = preprocess_data(
            sim_df, 
            sim_df['Current Moisture'], 
            sim_df['Current N'], 
            sim_df['Current P'], 
            sim_df['Current K']
        )

        # Define preprocessing pipeline
        self.preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Soil Type", "Crop", "Stage"]),
            ('num', StandardScaler(), [
                "Duration", "Ideal Moisture", "Max Temp", "Min Temp",
                "Water Retention", "Current Moisture", "Current N", 
                "Current P", "Current K", "Moisture Deficit",
                "N Deficit", "P Deficit", "K Deficit"
            ])
        ])

        # Prepare features and targets
        X = sim_df[["Soil Type", "Crop", "Stage"] + [
            "Duration", "Ideal Moisture", "Max Temp", "Min Temp",
            "Water Retention", "Current Moisture", "Current N",
            "Current P", "Current K", "Moisture Deficit", 
            "N Deficit", "P Deficit", "K Deficit"
        ]]

        y_irrigation = sim_df['Irrigation Needed']
        y_fertilizer = sim_df[['N Needed', 'P Needed', 'K Needed']]
        y_amount = sim_df['Moisture Deficit'] * sim_df['Water Retention']

        # Train models
        self.irrigation_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]).fit(X, y_irrigation)

        self.fertilizer_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]).fit(X, y_fertilizer)

        self.irrigation_amount_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ]).fit(X, y_amount)

    def get_recommendations(self, field_data, weather_forecast):
        """Get comprehensive irrigation and fertilizer recommendations"""
        try:
            # Get crop requirements with error handling
            try:
                requirements = self.crop_reqs.get_requirements(
                    field_data["Soil Type"],
                    field_data["Crop"],
                    field_data["Stage"]
                )
            except ValueError as e:
                return {
                    'metadata': {
                        'success': False,
                        'error': str(e),
                        'original_request': {
                            'soil_type': field_data["Soil Type"],
                            'crop': field_data["Crop"],
                            'stage': field_data["Stage"]
                        }
                    },
                    'available_options': self.crop_reqs._get_available_combinations()
                }

            # Prepare response structure
            response = {
                'metadata': {
                    'original_request': {
                        'soil_type': field_data["Soil Type"],
                        'crop': field_data["Crop"],
                        'stage': field_data["Stage"]
                    },
                    'actual_match': {
                        'soil_type': requirements["Soil Type"],
                        'crop': requirements["Crop"],
                        'stage': requirements["Stage"]
                    },
                    'is_approximate': requirements.get('IsApproximate', False),
                    'notes': requirements.get('Notes', ''),
                    'success': True
                },
                'recommendations': {}
            }

            # Prepare input data for prediction
            complete_data = {
                **requirements,
                "Current Moisture": field_data["Current Moisture"],
                "Current N": field_data["Current N"],
                "Current P": field_data["Current P"],
                "Current K": field_data["Current K"]
            }
            
            input_df = preprocess_data(
                pd.DataFrame([complete_data]), 
                complete_data["Current Moisture"],
                complete_data["Current N"],
                complete_data["Current P"],
                complete_data["Current K"]
            )
            
            # Make predictions
            upcoming_rain = self._check_upcoming_rain(weather_forecast)
            irrigation_needed = self.irrigation_model.predict(input_df)[0]
            fertilizer_needs = self.fertilizer_model.predict(input_df)[0]
            irrigation_amount = self.irrigation_amount_model.predict(input_df)[0]
            
            # Build irrigation recommendations
            response['recommendations']['irrigation'] = {
                "current": field_data["Current Moisture"],
                "ideal": requirements["Ideal Moisture"],
                "status": self._get_moisture_status(
                    field_data["Current Moisture"],
                    requirements["Ideal Moisture"]
                ),
                "needed": bool(irrigation_needed),
                "amount": round(max(0, irrigation_amount), 1),
                "upcoming_rain": upcoming_rain,
                "advice": self._get_irrigation_final_recommendation(
                    irrigation_needed, 
                    irrigation_amount, 
                    upcoming_rain
                ),
                "details": {
                    "water_retention": requirements["Water Retention"],
                    "rainfall_tolerance": requirements["Rainfall Tolerance"],
                    "irrigation_frequency": requirements["Irrigation Freq"]
                }
            }
            
            # Build fertilizer recommendations
            response['recommendations']['fertilizer'] = {
                "N": self._get_nutrient_recommendation('N', fertilizer_needs[0], field_data["Current N"], requirements["N"], input_df.iloc[0], requirements["Fertilizer"]),
                "P": self._get_nutrient_recommendation('P', fertilizer_needs[1], field_data["Current P"], requirements["P"], input_df.iloc[0], requirements["Fertilizer"]),
                "K": self._get_nutrient_recommendation('K', fertilizer_needs[2], field_data["Current K"], requirements["K"], input_df.iloc[0], requirements["Fertilizer"]),
                "general_advice": requirements.get("Notes", "")
            }
            
            return response

        except Exception as e:
            return {
                'metadata': {
                    'success': False,
                    'error': f"System error: {str(e)}",
                    'original_request': {
                        'soil_type': field_data.get("Soil Type", ""),
                        'crop': field_data.get("Crop", ""),
                        'stage': field_data.get("Stage", "")
                    }
                },
                'available_options': self.crop_reqs._get_available_combinations()
            }

    def _get_nutrient_recommendation(self, nutrient, needed, current, ideal, input_data, fertilizer_type):
        """Generate complete nutrient recommendation structure"""
        return {
            "needed": bool(needed),
            "current": current,
            "ideal": ideal,
            "status": self._get_npk_status(current, ideal, nutrient),
            "recommendation": self._get_single_fertilizer_recommendation(
                nutrient, 
                needed, 
                input_data
            ),
            "fertilizer_type": fertilizer_type,
            "deficit": input_data.get(f"{nutrient} Deficit", 0)
        }

    def _check_upcoming_rain(self, weather_forecast):
        """Check if significant rain is expected in the next 24 hours"""
        if not weather_forecast:
            return False
        
        # Sum expected rainfall in next 24 hours
        total_rain = sum(float(hour.get("Rain Amount (mm)", 0)) for hour in weather_forecast[:24])
        return total_rain > 5  # Consider significant if >5mm

    def _get_moisture_status(self, current, ideal):
        """Determine moisture status description"""
        if current >= ideal:
            return "Adequate"
        elif ideal - current <= 10:
            return "Slightly low"
        elif ideal - current <= 20:
            return "Moderately low"
        else:
            return "Critically low"

    def _get_npk_status(self, current, ideal, nutrient_type):
        """Determine NPK status with thresholds based on nutrient type"""
        threshold_map = {'N': 5, 'P': 3, 'K': 2}
        threshold = threshold_map.get(nutrient_type, 3)
        
        if current >= ideal:
            return "Sufficient"
        elif ideal - current <= threshold:
            return "Slightly deficient"
        else:
            return "Deficient"

    def _get_single_fertilizer_recommendation(self, nutrient, needed, input_data):
        """Get recommendation for a single nutrient"""
        if not needed:
            return f"No {nutrient} fertilizer needed at this time"
        
        deficit = input_data.get(f"{nutrient} Deficit", 0)
        amount = max(0, deficit)
        return f"Apply {amount:.1f} kg/ha of {nutrient}-based fertilizer"

    def _get_irrigation_final_recommendation(self, needed, amount, upcoming_rain):
        """Generate final irrigation advice with context"""
        if not needed:
            return "No irrigation needed - moisture levels are adequate"
        
        if upcoming_rain:
            if amount >= 30:
                return (
                    f"Delay irrigation. Significant rain expected soon which may meet or exceed "
                    f"the recommended {amount:.1f}mm irrigation."
                )
            else:
                return (
                    f"Partial rain expected. Consider applying {amount:.1f}mm after rainfall "
                    f"if moisture remains insufficient."
                )
        
        return (
            f"Irrigate with {amount:.1f}mm of water. "
            f"Morning application recommended to reduce evaporation losses."
        )


def initialize_advisor():
    """Initialize the irrigation advisor with crop data"""
    data, columns = get_crop_data()
    df = pd.DataFrame(data, columns=columns)
    crop_reqs = CropRequirements(df)
    return IrrigationAdvisor(crop_reqs)


# Global advisor instance
advisor = initialize_advisor()


def get_recommendations(field_data: dict, weather_forecast: list) -> dict:
    """Public interface to get recommendations"""
    return advisor.get_recommendations(field_data, weather_forecast)