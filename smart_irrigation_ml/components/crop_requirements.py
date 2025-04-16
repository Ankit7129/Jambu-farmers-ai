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
