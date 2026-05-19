import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

class PriceModel:
    def __init__(self):
        """Initialize the price prediction model"""
        self.df = None
        self.model = None
        self.FEATURES = None
        self.le_region = LabelEncoder()
        self.le_commodity = LabelEncoder()
        self.mae = None
        
        self.load_data()
        self.load_model()
        
    def load_data(self):
        """Load and preprocess the price data"""
        # Check if data file exists
        if not os.path.exists('data/data.csv'):
            raise FileNotFoundError("data/data.csv not found. Please ensure the file exists.")
        
        self.df = pd.read_csv('data/data.csv')
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Create temporal features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        
        # Encode categorical variables
        # Always fit new encoders (don't try to load saved ones)
        self.df['region_encoded'] = self.le_region.fit_transform(self.df['region'])
        self.df['commodity_encoded'] = self.le_commodity.fit_transform(self.df['commodity'])
        
        # Feature engineering (lags, rolling stats)
        self.create_features()
        
        # Define features (adjust based on your actual FEATURES list)
        self.FEATURES = [
            'region_encoded', 'commodity_encoded',
            'year', 'month', 'week_of_year', 'day_of_week', 'is_weekend',
            'lag_1', 'lag_7', 'lag_14', 'lag_30',
            'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7',
            'demand_index', 'supply_index', 'market_pressure',
            'avg_temperature_c', 'rainfall_mm'
        ]
    
    def create_features(self):
        """Create lag and rolling features"""
        # Group by region and commodity
        groups = self.df.groupby(['region', 'commodity'])
        
        # Create lag features
        for lag in [1, 7, 14, 30]:
            self.df[f'lag_{lag}'] = groups['price'].shift(lag)
        
        # Rolling statistics
        self.df['rolling_mean_7'] = groups['price'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        ).shift(1)
        
        self.df['rolling_mean_30'] = groups['price'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        ).shift(1)
        
        self.df['rolling_std_7'] = groups['price'].transform(
            lambda x: x.rolling(7, min_periods=1).std()
        ).shift(1)
        
        # Market indices
        self.df['demand_index'] = self.df['price'] / self.df['rolling_mean_30'].replace(0, np.nan)
        self.df['supply_index'] = self.df['rolling_mean_7'] / self.df['rolling_mean_30'].replace(0, np.nan)
        self.df['market_pressure'] = self.df['price'] - self.df['rolling_mean_7']
        
        # Fill NaN values
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')
    
    def load_model(self):
        """Load the trained XGBoost model or create a simple one if not found"""
        try:
            if os.path.exists('data/market_price_model.pkl'):
                self.model = joblib.load('data/market_price_model.pkl')
                print("Loaded trained price model")
            else:
                # Create a simple model if trained model doesn't exist
                from xgboost import XGBRegressor
                print("Training new price model...")
                self.train_model()
                print("Price model trained successfully")
                
            # Load features if available
            if os.path.exists('data/model_features.pkl'):
                self.FEATURES = joblib.load('data/model_features.pkl')
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a simple fallback model
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=10, random_state=42)
            print("Created fallback model")
    
    def train_model(self):
        """Train the price prediction model"""
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        
        # Prepare features and target
        X = self.df[self.FEATURES]
        y = self.df['price']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Train model
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Save the model
        os.makedirs('data', exist_ok=True)
        joblib.dump(self.model, 'data/market_price_model.pkl')
        joblib.dump(self.FEATURES, 'data/model_features.pkl')
        
        # Calculate MAE
        from sklearn.metrics import mean_absolute_error
        preds = self.model.predict(X_test)
        self.mae = mean_absolute_error(y_test, preds)
        joblib.dump(self.mae, 'data/model_mae.pkl')
    
    def get_available_regions(self):
        """Get list of available regions"""
        return sorted(self.df['region'].unique().tolist())
    
    def get_available_commodities(self):
        """Get list of available commodities"""
        return sorted(self.df['commodity'].unique().tolist())
    
    def get_last_known_row(self, region, commodity):
        """Get the most recent data for a region-commodity pair"""
        data = self.df[
            (self.df['region'].str.lower() == region.lower()) &
            (self.df['commodity'].str.lower() == commodity.lower())
        ].sort_values('date')
        
        if data.empty:
            return None
        
        return data.iloc[-1].copy()
    
    def get_historical_price(self, region, commodity, target_date):
        """Get historical price if available"""
        try:
            target_date = pd.to_datetime(target_date)
            
            data = self.df[
                (self.df['region'].str.lower() == region.lower()) &
                (self.df['commodity'].str.lower() == commodity.lower())
            ].copy()
            
            if data.empty:
                return None
            
            data['date'] = pd.to_datetime(data['date'])
            row = data[data['date'] == target_date]
            
            if row.empty:
                return None
            
            return float(row.iloc[0]['price'])
        except:
            return None
    
    def get_historical_avg_weather(self, region, month):
        """Get average weather for a region in specific month"""
        try:
            data = self.df[
                self.df['region'].str.lower() == region.lower()
            ].copy()
            
            if data.empty:
                return 25.0, 50.0  # Default values
            
            avg_temp = data[data['month'] == month]['avg_temperature_c'].mean()
            avg_rain = data[data['month'] == month]['rainfall_mm'].mean()
            
            # Handle NaN values
            avg_temp = 25.0 if pd.isna(avg_temp) else avg_temp
            avg_rain = 50.0 if pd.isna(avg_rain) else avg_rain
            
            return round(avg_temp, 2), round(avg_rain, 2)
        except:
            return 25.0, 50.0  # Default values
    
    def predict_price(self, region, commodity, target_date, temp=None, rain=None):
        """
        Predict price for a given region, commodity, and date
        """
        try:
            # Validate inputs
            if not region or not commodity or not target_date:
                return {
                    'error': True,
                    'message': 'Region, commodity, and date are required'
                }
            
            target_date = pd.to_datetime(target_date)
            
            # Get available options
            available_regions = self.get_available_regions()
            available_commodities = self.get_available_commodities()
            
            # Create case-insensitive maps
            region_map = {r.lower(): r for r in available_regions}
            commodity_map = {c.lower(): c for c in available_commodities}
            
            region_lower = region.lower()
            commodity_lower = commodity.lower()
            
            # Validate region and commodity
            if region_lower not in region_map:
                return {
                    'error': True,
                    'message': f'Invalid region. Available: {", ".join(available_regions[:5])}...'
                }
            
            if commodity_lower not in commodity_map:
                return {
                    'error': True,
                    'message': f'Invalid commodity. Available: {", ".join(available_commodities[:5])}...'
                }
            
            # Get correct case versions
            region = region_map[region_lower]
            commodity = commodity_map[commodity_lower]
            
            # Check for historical price first
            historical_price = self.get_historical_price(region, commodity, target_date)
            
            if historical_price is not None:
                return {
                    'error': False,
                    'mode': 'HISTORICAL',
                    'region': region,
                    'commodity': commodity,
                    'date': target_date.strftime('%Y-%m-%d'),
                    'price': round(historical_price, 2),
                    'temperature': None,
                    'rainfall': None,
                    'message': f'Historical price for {target_date.strftime("%Y-%m-%d")}: ₹{historical_price:.2f}/kg'
                }
            
            # For future dates, use prediction
            last_row = self.get_last_known_row(region, commodity)
            
            if last_row is None:
                return {
                    'error': True,
                    'message': 'No data available for this region and commodity combination'
                }
            
            current_date = pd.to_datetime(last_row['date'])
            
            if target_date <= current_date:
                return {
                    'error': True,
                    'message': f'Date must be after the last available date: {current_date.strftime("%Y-%m-%d")}'
                }
            
            # Get expected weather
            if temp is None or rain is None:
                exp_temp, exp_rain = self.get_historical_avg_weather(region, target_date.month)
            else:
                exp_temp, exp_rain = float(temp), float(rain)
            
            # Prepare for iterative prediction
            current_row = last_row.copy()
            next_price = None
            
            while current_date < target_date:
                # Update weather with expected values
                current_row['avg_temperature_c'] = exp_temp
                current_row['rainfall_mm'] = exp_rain
                
                # Prepare features for prediction
                feature_values = []
                for feature in self.FEATURES:
                    if feature in current_row:
                        feature_values.append(current_row[feature])
                    else:
                        # Default value for missing features
                        feature_values.append(0)
                
                # Predict price
                X = pd.DataFrame([feature_values], columns=self.FEATURES)
                next_price = float(self.model.predict(X)[0])
                
                # Update lag features (simplified)
                current_row['lag_30'] = current_row['lag_14'] if 'lag_14' in current_row else next_price
                current_row['lag_14'] = current_row['lag_7'] if 'lag_7' in current_row else next_price
                current_row['lag_7'] = current_row['lag_1'] if 'lag_1' in current_row else next_price
                current_row['lag_1'] = next_price
                
                # Move to next day
                current_date += pd.Timedelta(days=1)
                current_row['date'] = current_date
            
            if next_price is None:
                return {
                    'error': True,
                    'message': 'Could not generate prediction'
                }
            
            return {
                'error': False,
                'mode': 'PREDICTED',
                'region': region,
                'commodity': commodity,
                'date': target_date.strftime('%Y-%m-%d'),
                'price': round(next_price, 2),
                'temperature': exp_temp,
                'rainfall': exp_rain,
                'message': f'Predicted price for {target_date.strftime("%Y-%m-%d")}: ₹{next_price:.2f}/kg (Forecast based on historical trends)'
            }
            
        except Exception as e:
            return {
                'error': True,
                'message': f'Prediction error: {str(e)}'
            }