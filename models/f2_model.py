import pandas as pd
import pickle
import json
import os
from sklearn.preprocessing import LabelEncoder

class F2Model:
    def __init__(self):
        """Initialize the F2 model"""
        self.model = None
        self.encode_soil = None
        self.encode_crop = None
        self.soil_map = None
        self.crop_map = None
        self.fert_map = None
        
        self.load_model()
        self.load_data()
        
    def load_model(self):
        """Load the trained model"""
        self.model = pickle.load(open('data/classifier.pkl', 'rb'))
    
    def load_data(self):
        """Load and prepare data"""
        self.data = pd.read_csv('data/f2.csv')
        self.data.rename(columns={
            'Humidity ': 'Humidity',
            'Soil Type': 'Soil_Type',
            'Crop Type': 'Crop_Type',
            'Fertilizer Name': 'Fertilizer'
        }, inplace=True)
        
        # Initialize encoders
        self.encode_soil = LabelEncoder()
        self.encode_soil.fit(self.data['Soil_Type'])
        
        self.encode_crop = LabelEncoder()
        self.encode_crop.fit(self.data['Crop_Type'])
        
        # Create mappings for case-insensitive input
        self.soil_map = {cls.lower(): i for i, cls in enumerate(self.encode_soil.classes_)}
        self.crop_map = {cls.lower(): i for i, cls in enumerate(self.encode_crop.classes_)}
        
        # Fertilizer mapping
        self.fert_map = {
            0: "10-10-10", 1: "10-26-26", 2: "14-14-14", 3: "14-35-14",
            4: "15-15-15", 5: "17-17-17", 6: "20-20", 7: "28-28",
            8: "DAP", 9: "Potassium chloride", 10: "Potassium sulfate",
            11: "Superphosphate", 12: "TSP", 13: "Urea"
        }
    
    def get_soil_types(self):
        """Get available soil types"""
        return list(self.encode_soil.classes_)
    
    def get_crop_types(self):
        """Get available crop types"""
        return list(self.encode_crop.classes_)
    
    def predict(self, input_data):
        """
        Predict fertilizer recommendation
        """
        # Validate inputs
        soil_type_lower = input_data['soil_type'].lower()
        crop_type_lower = input_data['crop_type'].lower()
        
        if soil_type_lower not in self.soil_map:
            return {'error': True, 'message': 'Invalid soil type'}
        
        if crop_type_lower not in self.crop_map:
            return {'error': True, 'message': 'Invalid crop type'}
        
        # Encode soil and crop types
        soil_encoded = self.soil_map[soil_type_lower]
        crop_encoded = self.crop_map[crop_type_lower]
        
        # Prepare input array
        prediction_input = [[
            float(input_data['temperature']),
            float(input_data['humidity']),
            float(input_data['moisture']),
            soil_encoded,
            crop_encoded,
            int(input_data['nitrogen']),
            int(input_data['potassium']),
            int(input_data['phosphorous'])
        ]]
        
        # Make prediction
        pred = self.model.predict(prediction_input)[0]
        fertilizer = self.fert_map[pred]
        
        return {
            'error': False,
            'message': f"Recommended Fertilizer: {fertilizer}"
        }