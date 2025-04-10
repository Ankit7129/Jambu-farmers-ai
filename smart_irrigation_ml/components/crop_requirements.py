# components/crop_requirements.py

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
