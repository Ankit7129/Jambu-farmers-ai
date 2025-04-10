# smart_irrigation_ml/crop_dataset.py

def get_crop_data():
    data = [
        # Loamy/Clayey Soil - Paddy (Corrected values)
        ["Loamy/Clayey", "Paddy", "Nursery", 21, 85, 35, 22, "DAP", 20, 20, 10, "1-2", "50-100", 2, "Transplant ready at 3-leaf stage"],
        ["Loamy/Clayey", "Paddy", "Transplanting", 25, 85, 34, 24, "Urea + DAP", 40, 20, 10, "Continuous", "100-150", 3, "Maintain 5cm standing water"],
        ["Loamy/Clayey", "Paddy", "Tillering", 30, 75, 32, 20, "Urea", 30, 10, 20, "3-4", "50-80", 2, "Encourages shoot multiplication"],
        ["Loamy/Clayey", "Paddy", "Panicle Initiation", 20, 70, 30, 20, "DAP + MOP", 20, 30, 30, "3-4", "30-50", 2, "Critical yield stage"],
        ["Loamy/Clayey", "Paddy", "Flowering", 15, 70, 29, 20, "2% Urea Spray", 10, 10, 10, "2", "40", 2, "Avoid water stress"],
        ["Loamy/Clayey", "Paddy", "Grain Filling", 20, 60, 28, 18, "Potash", 5, 10, 25, "4-5", "50-80", 2, "Grain size development"],
        ["Loamy/Clayey", "Paddy", "Maturity", 15, 50, 27, 16, "None", 0, 0, 0, "Stop", "<20", 1, "Drain field completely"],

        # Alluvial Soil - Paddy (Corrected values)
        ["Alluvial", "Paddy", "Nursery", 21, 85, 35, 22, "DAP", 20, 20, 10, "1-2", "60-110", 2, "Transplant ready at 3-leaf stage"],
        ["Alluvial", "Paddy", "Transplanting", 25, 85, 34, 24, "Urea + DAP", 40, 20, 10, "Continuous", "120-160", 3, "Maintain 5cm standing water"],
        ["Alluvial", "Paddy", "Tillering", 30, 75, 32, 20, "Urea", 30, 10, 20, "3-4", "60-90", 2, "Encourages shoot multiplication"],
        ["Alluvial", "Paddy", "Panicle Initiation", 20, 70, 30, 20, "DAP + MOP", 20, 30, 30, "3-4", "40-60", 2, "Critical yield stage"],
        ["Alluvial", "Paddy", "Flowering", 15, 70, 29, 20, "2% Urea Spray", 10, 10, 10, "2", "50", 2, "Avoid water stress"],
        ["Alluvial", "Paddy", "Grain Filling", 20, 60, 28, 18, "Potash", 5, 10, 25, "4-5", "50-80", 2, "Grain size development"],
        ["Alluvial", "Paddy", "Maturity", 15, 50, 27, 16, "None", 0, 0, 0, "Stop", "<30", 1, "Drain field completely"],

        # Loamy Soil - Wheat
        ["Loamy", "Wheat", "Germination", 10, 70, 25, 18, "DAP", 20, 20, 0, "7-10", "30-50", 2, "Maintain moist topsoil"],
        ["Loamy", "Wheat", "Tillering", 30, 65, 22, 15, "Urea", 50, 0, 0, "14-21", "40-60", 2, "Critical for shoot count"],
        ["Loamy", "Wheat", "Stem Extension", 35, 70, 25, 12, "Urea + MOP", 30, 0, 20, "10-14", "50-70", 2, "Avoid water logging"],
        ["Loamy", "Wheat", "Heading", 15, 65, 28, 15, "NPK", 20, 20, 20, "10", "30-40", 1, "Sensitive to drought"],
        ["Loamy", "Wheat", "Grain Filling", 30, 60, 30, 18, "Potash", 0, 0, 25, "12-15", "40-50", 1, "Reduce irrigation gradually"],
        
        # Alluvial Soil - Maize
        ["Alluvial", "Maize", "Germination", 10, 75, 32, 20, "DAP", 20, 20, 0, "5-7", "40-60", 2, "Uniform moisture critical"],
        ["Alluvial", "Maize", "Vegetative", 35, 70, 35, 22, "Urea", 60, 0, 0, "7-10", "60-80", 2, "Rapid growth phase"],
        ["Alluvial", "Maize", "Tasseling", 15, 75, 33, 20, "NPK", 30, 20, 20, "5-7", "50-70", 3, "Most water-sensitive stage"],
        ["Alluvial", "Maize", "Silking", 20, 80, 32, 21, "Urea Spray", 15, 0, 0, "5", "60-80", 3, "Irrigate if no rain"],
        ["Alluvial", "Maize", "Grain Development", 40, 70, 30, 18, "Potash", 0, 0, 30, "10-12", "50-70", 2, "Reduce water at maturity"]
    ]

    columns = [
        "Soil Type", "Crop", "Stage", "Duration", "Ideal Moisture",
        "Max Temp", "Min Temp", "Fertilizer", "N", "P", "K",
        "Irrigation Freq (days)", "Rainfall Tolerance (mm)", "Water Retention", "Notes"
    ]

    return data, columns