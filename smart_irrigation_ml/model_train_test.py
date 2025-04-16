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
        # Loamy/Clayey Soil - Paddy (Complete growth cycle)
        ["Loamy/Clayey", "Paddy", "Nursery", 21, 85, 35, 22, "DAP", 20, 20, 10, "1-2", "50-100", 2, "Transplant ready at 3-leaf stage"],
        ["Loamy/Clayey", "Paddy", "Land Preparation", 7, 90, 34, 23, "Farmyard Manure", 0, 0, 0, "Flooding", "100-150", 3, "Puddle the field"],
        ["Loamy/Clayey", "Paddy", "Transplanting", 25, 85, 34, 24, "Urea + DAP", 40, 20, 10, "Continuous", "100-150", 3, "Maintain 5cm standing water"],
        ["Loamy/Clayey", "Paddy", "Tillering", 30, 75, 32, 20, "Urea", 30, 10, 20, "3-4", "50-80", 2, "Encourages shoot multiplication"],
        ["Loamy/Clayey", "Paddy", "Stem Elongation", 15, 80, 33, 21, "NPK", 25, 25, 25, "4-5", "60-90", 2, "Critical growth phase"],
        ["Loamy/Clayey", "Paddy", "Panicle Initiation", 20, 70, 30, 20, "DAP + MOP", 20, 30, 30, "3-4", "30-50", 2, "Critical yield stage"],
        ["Loamy/Clayey", "Paddy", "Flowering", 15, 70, 29, 20, "2% Urea Spray", 10, 10, 10, "2", "40", 2, "Avoid water stress"],
        ["Loamy/Clayey", "Paddy", "Grain Filling", 20, 60, 28, 18, "Potash", 5, 10, 25, "4-5", "50-80", 2, "Grain size development"],
        ["Loamy/Clayey", "Paddy", "Maturity", 15, 50, 27, 16, "None", 0, 0, 0, "Stop", "<20", 1, "Drain field completely"],
        ["Loamy/Clayey", "Paddy", "Harvest", 7, 40, 30, 15, "None", 0, 0, 0, "None", "Any", 0, "Grains at 20-22% moisture"],

        # Alluvial Soil - Paddy (Complete growth cycle)
        ["Alluvial", "Paddy", "Nursery", 21, 85, 35, 22, "DAP", 20, 20, 10, "1-2", "60-110", 2, "Transplant ready at 3-leaf stage"],
        ["Alluvial", "Paddy", "Land Preparation", 7, 90, 34, 23, "Compost", 0, 5, 5, "Flooding", "120-170", 3, "Puddle the field thoroughly"],
        ["Alluvial", "Paddy", "Transplanting", 25, 85, 34, 24, "Urea + DAP", 40, 20, 10, "Continuous", "120-160", 3, "Maintain 5cm standing water"],
        ["Alluvial", "Paddy", "Tillering", 30, 75, 32, 20, "Urea", 30, 10, 20, "3-4", "60-90", 2, "Encourages shoot multiplication"],
        ["Alluvial", "Paddy", "Stem Elongation", 15, 80, 33, 21, "NPK", 25, 25, 25, "4-5", "70-100", 2, "Critical growth phase"],
        ["Alluvial", "Paddy", "Panicle Initiation", 20, 70, 30, 20, "DAP + MOP", 20, 30, 30, "3-4", "40-60", 2, "Critical yield stage"],
        ["Alluvial", "Paddy", "Flowering", 15, 70, 29, 20, "2% Urea Spray", 10, 10, 10, "2", "50", 2, "Avoid water stress"],
        ["Alluvial", "Paddy", "Grain Filling", 20, 60, 28, 18, "Potash", 5, 10, 25, "4-5", "50-80", 2, "Grain size development"],
        ["Alluvial", "Paddy", "Maturity", 15, 50, 27, 16, "None", 0, 0, 0, "Stop", "<30", 1, "Drain field completely"],
        ["Alluvial", "Paddy", "Harvest", 7, 40, 30, 15, "None", 0, 0, 0, "None", "Any", 0, "Grains at 20-22% moisture"],

        # Loamy Soil - Wheat (Complete growth cycle)
        ["Loamy", "Wheat", "Land Preparation", 14, 60, 25, 15, "Farmyard Manure", 0, 0, 0, "Pre-sowing", "30-50", 2, "Fine tilth required"],
        ["Loamy", "Wheat", "Germination", 10, 70, 25, 18, "DAP", 20, 20, 0, "7-10", "30-50", 2, "Maintain moist topsoil"],
        ["Loamy", "Wheat", "Tillering", 30, 65, 22, 15, "Urea", 50, 0, 0, "14-21", "40-60", 2, "Critical for shoot count"],
        ["Loamy", "Wheat", "Stem Extension", 35, 70, 25, 12, "Urea + MOP", 30, 0, 20, "10-14", "50-70", 2, "Avoid water logging"],
        ["Loamy", "Wheat", "Booting", 10, 75, 27, 14, "NPK", 20, 20, 20, "7-10", "40-50", 2, "Head formation begins"],
        ["Loamy", "Wheat", "Heading", 15, 65, 28, 15, "NPK", 20, 20, 20, "10", "30-40", 1, "Sensitive to drought"],
        ["Loamy", "Wheat", "Flowering", 10, 70, 30, 16, "Boron Spray", 0, 0, 0, "7", "20-30", 1, "Critical pollination stage"],
        ["Loamy", "Wheat", "Grain Filling", 30, 60, 30, 18, "Potash", 0, 0, 25, "12-15", "40-50", 1, "Reduce irrigation gradually"],
        ["Loamy", "Wheat", "Maturity", 20, 50, 32, 20, "None", 0, 0, 0, "Stop", "<20", 0, "Grains hard, moisture <15%"],
        ["Loamy", "Wheat", "Harvest", 7, 40, 35, 22, "None", 0, 0, 0, "None", "Any", 0, "Timely harvest prevents shattering"],

        # Black Soil - Cotton (Complete growth cycle)
        ["Black", "Cotton", "Land Preparation", 21, 60, 35, 25, "Compost", 0, 0, 0, "Pre-sowing", "50-70", 2, "Deep ploughing beneficial"],
        ["Black", "Cotton", "Germination", 14, 70, 38, 26, "DAP", 20, 20, 0, "5-7", "40-60", 2, "Avoid crust formation"],
        ["Black", "Cotton", "Seedling", 21, 65, 40, 28, "Urea", 30, 0, 0, "7-10", "50-70", 2, "Critical establishment phase"],
        ["Black", "Cotton", "Square Formation", 35, 70, 38, 26, "Urea + MOP", 40, 0, 20, "10-14", "60-80", 2, "Bud development stage"],
        ["Black", "Cotton", "Flowering", 28, 75, 35, 24, "NPK", 30, 20, 30, "7-10", "70-90", 3, "Peak water requirement"],
        ["Black", "Cotton", "Boll Development", 42, 70, 33, 22, "Potash", 0, 0, 40, "14-21", "60-80", 2, "Most critical yield stage"],
        ["Black", "Cotton", "Boll Maturation", 35, 60, 32, 20, "None", 0, 0, 0, "21", "30-50", 1, "Reduce irrigation"],
        ["Black", "Cotton", "Harvest", 60, 50, 30, 18, "None", 0, 0, 0, "None", "Any", 0, "Staggered picking as bolls open"],

        # Sandy Loam - Sugarcane (Complete growth cycle)
        ["Sandy Loam", "Sugarcane", "Land Preparation", 30, 70, 35, 24, "Farmyard Manure", 0, 0, 0, "Pre-planting", "80-100", 2, "Deep ploughing required"],
        ["Sandy Loam", "Sugarcane", "Germination", 45, 75, 38, 26, "DAP", 30, 30, 0, "7-10", "100-120", 3, "Critical establishment phase"],
        ["Sandy Loam", "Sugarcane", "Tillering", 90, 80, 40, 28, "Urea", 60, 0, 0, "10-14", "120-150", 3, "Maximize shoot population"],
        ["Sandy Loam", "Sugarcane", "Grand Growth", 120, 85, 42, 30, "NPK", 80, 40, 40, "7-10", "150-200", 3, "Peak growth period"],
        ["Sandy Loam", "Sugarcane", "Maturity", 60, 70, 35, 24, "Potash", 0, 0, 50, "14-21", "100-120", 2, "Sugar accumulation phase"],
        ["Sandy Loam", "Sugarcane", "Ripening", 30, 60, 32, 22, "None", 0, 0, 0, "21", "50-70", 1, "Withhold irrigation"],
        ["Sandy Loam", "Sugarcane", "Harvest", 7, 50, 30, 20, "None", 0, 0, 0, "None", "Any", 0, "Optimum sucrose content"],

        # Red Soil - Groundnut (Complete growth cycle)
        ["Red", "Groundnut", "Land Preparation", 14, 65, 35, 25, "Compost", 0, 0, 0, "Pre-sowing", "40-60", 2, "Fine tilth important"],
        ["Red", "Groundnut", "Germination", 10, 70, 38, 26, "DAP", 20, 20, 0, "5-7", "50-70", 2, "Avoid water logging"],
        ["Red", "Groundnut", "Vegetative", 30, 75, 40, 28, "Urea", 30, 0, 0, "7-10", "60-80", 2, "Foliage development"],
        ["Red", "Groundnut", "Flowering", 20, 80, 38, 26, "NPK", 30, 30, 30, "5-7", "70-90", 2, "Peg formation begins"],
        ["Red", "Groundnut", "Pegging", 30, 80, 35, 24, "Gypsum", 0, 30, 0, "7", "80-100", 2, "Critical calcium requirement"],
        ["Red", "Groundnut", "Pod Development", 40, 75, 33, 22, "Potash", 0, 0, 40, "10-14", "70-90", 2, "Most water-sensitive stage"],
        ["Red", "Groundnut", "Maturity", 20, 60, 32, 20, "None", 0, 0, 0, "Stop", "30-50", 1, "Reduce irrigation"],
        ["Red", "Groundnut", "Harvest", 7, 50, 30, 18, "None", 0, 0, 0, "None", "Any", 0, "Pod moisture 25-30%"]
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
        self.similarity_threshold = 0.7  # Threshold for considering matches similar
        
    def get_requirements(self, soil_type, crop, stage):
        """Get ideal requirements for given soil type, crop and stage with flexible matching"""
        if not all([soil_type, crop, stage]):
            raise ValueError("All parameters (soil_type, crop, stage) must be provided")
            
        try:
            cache_key = f"{soil_type}_{crop}_{stage}"
            if cache_key in self.requirements_cache:
                return self.requirements_cache[cache_key]
                
            # Validate input types
            if not isinstance(soil_type, str) or not isinstance(crop, str) or not isinstance(stage, str):
                raise TypeError("All parameters must be strings")
            
            # Try exact match first
            exact_match = self._get_exact_match(soil_type, crop, stage)
            if exact_match:
                self.requirements_cache[cache_key] = exact_match
                return exact_match
                
            # If no exact match, find closest matches with warnings
            closest_match, match_type = self._find_closest_match(soil_type, crop, stage)
            
            if closest_match:
                # Add warning note about the approximation
                closest_match['Notes'] = f"Approximate recommendation ({match_type}). {closest_match.get('Notes', '')}"
                closest_match['IsApproximate'] = True
                closest_match['OriginalRequest'] = {
                    'soil_type': soil_type,
                    'crop': crop,
                    'stage': stage
                }
                self.requirements_cache[cache_key] = closest_match
                return closest_match
                
            # If nothing found at all
            available_combinations = self._get_available_combinations()
            raise ValueError(
                f"No data found for {soil_type}/{crop}/{stage} and no suitable approximations available.\n"
                f"Available soil types: {available_combinations['soils']}\n"
                f"Available crops: {available_combinations['crops']}\n"
                f"Available stages: {available_combinations['stages']}"
            )
            
        except Exception as e:
            # Log the error for debugging (in a real system you'd use logging)
            print(f"Error in get_requirements: {str(e)}")
            raise ValueError(
                f"Could not process request for {soil_type}/{crop}/{stage}. "
                f"Please verify your inputs and try again. Error: {str(e)}"
            )
    
    def _get_exact_match(self, soil_type, crop, stage):
        """Check for exact match in dataset"""
        try:
            mask = ((self.df['Soil Type'].str.lower() == soil_type.lower()) & 
                    (self.df['Crop'].str.lower() == crop.lower()) & 
                    (self.df['Stage'].str.lower() == stage.lower()))
            if any(mask):
                return self.df[mask].iloc[0].to_dict()
            return None
        except Exception as e:
            raise ValueError(f"Error searching for exact match: {str(e)}")
    
    def _find_closest_match(self, soil_type, crop, stage):
        """Find the closest matching record with flexible matching"""
        try:
            strategies = [
                ('similar_soil_crop_stage', self._match_similar_soil_crop_stage),
                ('same_crop_stage', self._match_same_crop_stage),
                ('same_crop_common_stage', self._match_same_crop_common_stage),
                ('similar_crop', self._match_similar_crop),
            ]
            
            for match_type, strategy in strategies:
                try:
                    result = strategy(soil_type, crop, stage)
                    if result:
                        return result, match_type
                except Exception as e:
                    print(f"Warning: Strategy {match_type} failed: {str(e)}")
                    continue
                    
            return None, None
        except Exception as e:
            raise ValueError(f"Error in closest match search: {str(e)}")
    
    def _match_similar_soil_crop_stage(self, soil_type, crop, stage):
        """Match with similar soil type (if soil is the only mismatch)"""
        try:
            crop_stage_mask = ((self.df['Crop'].str.lower() == crop.lower()) & 
                             (self.df['Stage'].str.lower() == stage.lower()))
            
            if any(crop_stage_mask):
                available_soils = self.df[crop_stage_mask]['Soil Type'].unique()
                closest_soil = self._find_closest_soil(soil_type, available_soils)
                
                if closest_soil:
                    mask = crop_stage_mask & (self.df['Soil Type'] == closest_soil)
                    return self.df[mask].iloc[0].to_dict()
            return None
        except Exception as e:
            raise ValueError(f"Error in similar soil matching: {str(e)}")
    
    def _match_same_crop_stage(self, soil_type, crop, stage):
        """Match same crop and stage, any soil"""
        try:
            mask = ((self.df['Crop'].str.lower() == crop.lower()) & 
                   (self.df['Stage'].str.lower() == stage.lower()))
            if any(mask):
                return self.df[mask].iloc[0].to_dict()
            return None
        except Exception as e:
            raise ValueError(f"Error in same crop/stage matching: {str(e)}")
    
    def _match_same_crop_common_stage(self, soil_type, crop, stage):
        """Match same crop with its most common stage"""
        try:
            mask = (self.df['Crop'].str.lower() == crop.lower())
            if any(mask):
                available_stages = self.df[mask]['Stage'].unique()
                closest_stage = self._find_closest_stage(stage, available_stages)
                
                if closest_stage:
                    mask = mask & (self.df['Stage'].str.lower() == closest_stage.lower())
                    return self.df[mask].iloc[0].to_dict()
            return None
        except Exception as e:
            raise ValueError(f"Error in same crop/common stage matching: {str(e)}")
    
    def _match_similar_crop(self, soil_type, crop, stage):
        """Match with similar crop (last resort)"""
        try:
            available_crops = self.df['Crop'].unique()
            closest_crop = self._find_closest_crop(crop, available_crops)
            
            if closest_crop:
                return self._match_same_crop_common_stage(soil_type, closest_crop, stage)
            return None
        except Exception as e:
            raise ValueError(f"Error in similar crop matching: {str(e)}")
    
    def _find_closest_soil(self, target_soil, available_soils):
        """Find most similar soil type using string similarity"""
        try:
            from difflib import get_close_matches
            matches = get_close_matches(
                target_soil.lower(),
                [s.lower() for s in available_soils],
                n=1,
                cutoff=self.similarity_threshold
            )
            if matches:
                # Return original case version
                return next(s for s in available_soils if s.lower() == matches[0])
            return None
        except Exception as e:
            raise ValueError(f"Error finding closest soil: {str(e)}")
    
    def _find_closest_stage(self, target_stage, available_stages):
        """Find most similar growth stage"""
        try:
            from difflib import get_close_matches
            matches = get_close_matches(
                target_stage.lower(),
                [s.lower() for s in available_stages],
                n=1,
                cutoff=self.similarity_threshold
            )
            if matches:
                # Return original case version
                return next(s for s in available_stages if s.lower() == matches[0])
            return None
        except Exception as e:
            raise ValueError(f"Error finding closest stage: {str(e)}")
    
    def _find_closest_crop(self, target_crop, available_crops):
        """Find most similar crop"""
        try:
            from difflib import get_close_matches
            matches = get_close_matches(
                target_crop.lower(),
                [c.lower() for c in available_crops],
                n=1,
                cutoff=self.similarity_threshold
            )
            if matches:
                # Return original case version
                return next(c for c in available_crops if c.lower() == matches[0])
            return None
        except Exception as e:
            raise ValueError(f"Error finding closest crop: {str(e)}")
    
    def _get_available_combinations(self):
        """Get lists of available options for helpful error messages"""
        return {
            'soils': sorted(self.df['Soil Type'].unique()),
            'crops': sorted(self.df['Crop'].unique()),
            'stages': sorted(self.df['Stage'].unique())
        }

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
    "Soil Type": "R",
    "Crop": "Gr",
    "Stage": "kmk",
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
print(f"Final advice: {recommendations['irrigation']['final_recommendation']}")

print("\n=== Fertilizer Recommendation ===")
print(f"Current N: {field_data['Current N']} (Ideal: {recommendations['fertilizer']['ideal_N']})")
print(f"Current P: {field_data['Current P']} (Ideal: {recommendations['fertilizer']['ideal_P']})")
print(f"Current K: {field_data['Current K']} (Ideal: {recommendations['fertilizer']['ideal_K']})")
for rec in recommendations['fertilizer']['recommendations']:
    print("-", rec)