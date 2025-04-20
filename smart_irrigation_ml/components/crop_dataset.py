def get_crop_data():
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

    return data, columns