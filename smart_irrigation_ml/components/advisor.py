import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .crop_dataset import get_crop_data
from .crop_requirements import CropRequirements


def preprocess_data(df, current_moisture, current_n, current_p, current_k):
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
        self.crop_reqs = crop_reqs
        self._train_models()

    def _train_models(self):
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
            sim_df, sim_df['Current Moisture'], sim_df['Current N'], sim_df['Current P'], sim_df['Current K']
        )

        self.preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Soil Type", "Crop", "Stage"]),
            ('num', StandardScaler(), [
                "Duration", "Ideal Moisture", "Max Temp", "Min Temp",
                "Water Retention", "Current Moisture", "Current N", 
                "Current P", "Current K", "Moisture Deficit",
                "N Deficit", "P Deficit", "K Deficit"
            ])
        ])

        X = sim_df[["Soil Type", "Crop", "Stage"] + [
            "Duration", "Ideal Moisture", "Max Temp", "Min Temp",
            "Water Retention", "Current Moisture", "Current N",
            "Current P", "Current K", "Moisture Deficit", "N Deficit", "P Deficit", "K Deficit"
        ]]

        y_irrigation = sim_df['Irrigation Needed']
        y_fertilizer = sim_df[['N Needed', 'P Needed', 'K Needed']]
        y_amount = sim_df['Moisture Deficit'] * sim_df['Water Retention']

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
        requirements = self.crop_reqs.get_requirements(
            field_data["Soil Type"], field_data["Crop"], field_data["Stage"]
        )

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

        upcoming_rain = self._check_upcoming_rain(weather_forecast)
        irrigation_needed = self.irrigation_model.predict(input_df)[0]
        fertilizer_needs = self.fertilizer_model.predict(input_df)[0]
        irrigation_amount = self.irrigation_amount_model.predict(input_df)[0]

        return {
            "irrigation": {
                "current_moisture": complete_data["Current Moisture"],
                "ideal_moisture": requirements["Ideal Moisture"],
                "needed": bool(irrigation_needed),
                "recommended_amount": round(max(0, irrigation_amount), 1),
                "consider_rain": upcoming_rain,
                "final_recommendation": self._get_irrigation_final_recommendation(
                    irrigation_needed, irrigation_amount, upcoming_rain
                )
            },
            "fertilizer": {
                "N": {
                    "needed": bool(fertilizer_needs[0]),
                    "ideal": requirements["N"],
                     "current": field_data["Current N"],
                    "status": self._get_npk_status(field_data["Current N"], requirements["N"], "N"),
                    "recommendations": self._get_fertilizer_recommendations('N', fertilizer_needs[0], input_df.iloc[0])
                },
                "P": {
                    "needed": bool(fertilizer_needs[1]),
                    "ideal": requirements["P"],
                     "current": field_data["Current P"],
                    "status": self._get_npk_status(field_data["Current P"], requirements["N"], "N"),
                    "recommendations": self._get_fertilizer_recommendations('P', fertilizer_needs[1], input_df.iloc[0])
                },
                "K": {
                    "needed": bool(fertilizer_needs[2]),
                    "ideal": requirements["K"],
                    "current": field_data["Current K"],
                    "status": self._get_npk_status(field_data["Current K"], requirements["N"], "N"),
                    "recommendations": self._get_fertilizer_recommendations('K', fertilizer_needs[2], input_df.iloc[0])
                }
            }
        }

    def _check_upcoming_rain(self, weather_forecast):
        if not weather_forecast:
            return False
        total_rain = sum(float(hour.get("Rain Amount (mm)", 0)) for hour in weather_forecast[:24])
        return total_rain > 5

    def _get_irrigation_final_recommendation(self, needed, amount, upcoming_rain):
        if not needed:
            return "No irrigation needed at this time based on current moisture and crop stage."

        if upcoming_rain:
            if amount >= 30:
                return (
                    f"Delay irrigation. Significant rain is expected, which may meet or exceed the required "
                    f"{amount:.1f}mm irrigation."
                )
            else:
                return (
                    f"Partial rain expected. Irrigate cautiously — expected rainfall may cover part of the "
                    f"{amount:.1f}mm recommended irrigation."
                )
        return f"Irrigate now with {amount:.1f}mm of water to meet the crop's needs."

    def _get_npk_status(self, current, ideal, nutrient):
        excess_ratio = current / ideal if ideal > 0 else 0
        if excess_ratio >= 1.5:
            return f"Warning: {nutrient} levels are too high and may harm the crop."
        elif excess_ratio >= 1.2:
            return f"Note: {nutrient} levels are slightly above ideal."
        elif excess_ratio <= 0.5:
            return f"{nutrient} levels are critically low."
        else:
            return f"{nutrient} levels are within the safe range."


    def _get_fertilizer_recommendations(self, nutrient, need, input_data):
        if not need:
            return "No fertilizer needed."
        deficit = input_data.get(f"{nutrient} Deficit", 0)
        return f"Apply {max(0, deficit):.1f} kg/ha of {nutrient}-based fertilizer."

    

def initialize_advisor():
    data, columns = get_crop_data()
    df = pd.DataFrame(data, columns=columns)
    crop_reqs = CropRequirements(df)
    return IrrigationAdvisor(crop_reqs)


advisor = initialize_advisor()


def get_recommendations(field_data: dict, weather_forecast: list) -> dict:
    return advisor.get_recommendations(field_data, weather_forecast)
