import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class CropModel:
    def __init__(self):
        """Initialize the crop recommendation model"""
        self.df = None
        self.model = None
        self.scaler = None
        self.le = None
        self.FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        self.load_data()
        self.load_model()
    
    def load_data(self):
        """Load crop recommendation data"""
        try:
            if os.path.exists('data/Crop_recommendation.csv'):
                self.df = pd.read_csv('data/Crop_recommendation.csv')
                print(f"Loaded crop data with {len(self.df)} rows")
            else:
                print("Warning: Crop_recommendation.csv not found")
                self.df = pd.DataFrame(columns=self.FEATURES + ['label'])
        except Exception as e:
            print(f"Error loading crop data: {e}")
            self.df = pd.DataFrame(columns=self.FEATURES + ['label'])
    
    def load_model(self):
        """Load trained model, scaler, and label encoder"""
        try:
            # Try to load pretrained model
            if os.path.exists('data/cropmodel.pkl'):
                self.model = pickle.load(open('data/cropmodel.pkl', 'rb'))
                print("Loaded pretrained crop model")
            elif os.path.exists('data/model_logistic.pkl'):
                self.model = pickle.load(open('data/model_logistic.pkl', 'rb'))
                print("Loaded logistic crop model")
            else:
                print("Training new crop model...")
                self.train_model()
            
            # Load scaler
            if os.path.exists('data/minmaxscaler.pkl'):
                self.scaler = pickle.load(open('data/minmaxscaler.pkl', 'rb'))
                print("Loaded scaler")
            else:
                print("Creating new scaler...")
                self.scaler = MinMaxScaler()
                if len(self.df) > 0:
                    self.scaler.fit(self.df[self.FEATURES])
                    os.makedirs('data', exist_ok=True)
                    pickle.dump(self.scaler, open('data/minmaxscaler.pkl', 'wb'))
            
            # Load label encoder
            if os.path.exists('data/labelencoder.pkl'):
                self.le = pickle.load(open('data/labelencoder.pkl', 'rb'))
                print("Loaded label encoder")
            else:
                print("Creating new label encoder...")
                self.le = LabelEncoder()
                if len(self.df) > 0 and 'label' in self.df.columns:
                    self.le.fit(self.df['label'])
                    os.makedirs('data', exist_ok=True)
                    pickle.dump(self.le, open('data/labelencoder.pkl', 'wb'))
            
        except Exception as e:
            print(f"Error loading crop model components: {e}")
            self.model = None
            self.scaler = None
            self.le = None
    
    def train_model(self):
        """Train a new crop recommendation model"""
        try:
            if len(self.df) == 0:
                print("No data available for training")
                return
            
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            
            # Prepare features and target
            X = self.df[self.FEATURES]
            y = self.df['label']
            
            # Encode labels
            self.le = LabelEncoder()
            y_encoded = self.le.fit_transform(y)
            
            # Create and fit scaler
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train model
            self.model = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='auto', random_state=42)
            self.model.fit(X_train, y_train)
            
            # Save model components
            os.makedirs('data', exist_ok=True)
            pickle.dump(self.model, open('data/cropmodel.pkl', 'wb'))
            pickle.dump(self.scaler, open('data/minmaxscaler.pkl', 'wb'))
            pickle.dump(self.le, open('data/labelencoder.pkl', 'wb'))
            
            print(f"Crop model trained and saved. Available crops: {len(self.le.classes_)}")
            
        except Exception as e:
            print(f"Error training crop model: {e}")
    
    def get_available_crops(self):
        """Get list of available crop names"""
        if self.le is not None:
            return sorted(self.le.classes_.tolist())
        elif len(self.df) > 0 and 'label' in self.df.columns:
            return sorted(self.df['label'].unique().tolist())
        else:
            return []
    
    def get_feature_ranges(self):
        """Get min-max ranges for each feature for user guidance"""
        if len(self.df) == 0:
            return {}
        
        ranges = {}
        for feature in self.FEATURES:
            if feature in self.df.columns:
                ranges[feature] = {
                    'min': float(self.df[feature].min()),
                    'max': float(self.df[feature].max()),
                    'mean': float(self.df[feature].mean())
                }
        return ranges
    
    def predict(self, input_data):
        """
        Predict recommended crop based on input parameters
        
        Args:
            input_data: Dictionary with feature values
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Validate model components
            if self.model is None or self.scaler is None or self.le is None:
                return {
                    'error': True,
                    'message': 'Crop recommendation model not properly loaded'
                }
            
            # Prepare input array
            input_values = []
            missing_features = []
            
            for feature in self.FEATURES:
                if feature in input_data:
                    try:
                        value = float(input_data[feature])
                        input_values.append(value)
                    except:
                        return {
                            'error': True,
                            'message': f'Invalid value for {feature}: {input_data[feature]}'
                        }
                else:
                    missing_features.append(feature)
            
            if missing_features:
                return {
                    'error': True,
                    'message': f'Missing required features: {", ".join(missing_features)}'
                }
            
            # Create DataFrame and scale
            input_df = pd.DataFrame([input_values], columns=self.FEATURES)
            input_scaled = self.scaler.transform(input_df)
            
            # Predict
            pred_label = self.model.predict(input_scaled)[0]
            crop_name = self.le.inverse_transform([pred_label])[0]
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_scaled)[0]
                # Get top 3 recommendations
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_crops = []
                for idx in top_indices:
                    top_crops.append({
                        'crop': self.le.inverse_transform([idx])[0],
                        'probability': float(probabilities[idx])
                    })
            else:
                top_crops = [{'crop': crop_name, 'probability': 1.0}]
            
            return {
                'error': False,
                'recommended_crop': crop_name,
                'top_recommendations': top_crops,
                'input_features': input_data,
                'message': f'Recommended Crop: {crop_name}'
            }
            
        except Exception as e:
            return {
                'error': True,
                'message': f'Prediction error: {str(e)}'
            }
    
    def validate_input_range(self, feature, value):
        """Check if input value is within typical range"""
        if len(self.df) == 0 or feature not in self.df.columns:
            return True, "No validation data available"
        
        min_val = self.df[feature].min()
        max_val = self.df[feature].max()
        
        if value < min_val or value > max_val:
            return False, f'{feature} value {value} is outside typical range ({min_val:.1f} - {max_val:.1f})'
        
        return True, "Value is within typical range"