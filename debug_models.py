import os
import sys

print("🔍 Debugging Model Loading Issues")
print("=" * 50)

# Check Python version
print(f"Python Version: {sys.version}")
print(f"Current Directory: {os.getcwd()}")

# Check if data files exist
data_files = [
    'data/f2.csv',
    'data/classifier.pkl',
    'data/data.csv',
    'data/Crop_recommendation.csv',
    'data/cropmodel.pkl',
    'data/minmaxscaler.pkl',
    'data/labelencoder.pkl'
]

print("\n📁 Checking Data Files:")
for file in data_files:
    exists = os.path.exists(file)
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"{status}: {file}")

# Try importing models
print("\n🔧 Testing Model Imports:")
try:
    from models.f2_model import F2Model
    print("✅ F2Model import successful")
except Exception as e:
    print(f"❌ F2Model import failed: {e}")

try:
    from models.price_model import PriceModel
    print("✅ PriceModel import successful")
except Exception as e:
    print(f"❌ PriceModel import failed: {e}")

try:
    from models.crop_model import CropModel
    print("✅ CropModel import successful")
except Exception as e:
    print(f"❌ CropModel import failed: {e}")

# Test loading models
print("\n🚀 Testing Model Initialization:")

# Test F2 Model
print("\n1. Testing F2 Model:")
try:
    f2 = F2Model()
    print("✅ F2Model initialized")
    print(f"   - Soil types: {len(f2.get_soil_types())}")
    print(f"   - Crop types: {len(f2.get_crop_types())}")
except Exception as e:
    print(f"❌ F2Model initialization failed: {e}")

# Test Price Model
print("\n2. Testing Price Model:")
try:
    price = PriceModel()
    print("✅ PriceModel initialized")
    print(f"   - Regions: {len(price.get_available_regions())}")
    print(f"   - Commodities: {len(price.get_available_commodities())}")
except Exception as e:
    print(f"❌ PriceModel initialization failed: {e}")

# Test Crop Model
print("\n3. Testing Crop Model:")
try:
    crop = CropModel()
    print("✅ CropModel initialized")
    print(f"   - Available crops: {len(crop.get_available_crops())}")
    print(f"   - Model loaded: {crop.model is not None}")
    print(f"   - Scaler loaded: {crop.scaler is not None}")
    print(f"   - LabelEncoder loaded: {crop.le is not None}")
except Exception as e:
    print(f"❌ CropModel initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Debug completed. Check above for issues.")