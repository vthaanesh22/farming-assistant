from flask import Flask, request, jsonify, render_template
import sys
import os

app = Flask(__name__)

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🚀 Starting Agricultural AI System...")
print("=" * 50)

# Initialize models with debugging
try:
    print("🔄 Loading F2 Model...")
    from models.f2_model import F2Model
    f2_model = F2Model()
    print(f"✅ F2 Model loaded - Model: {f2_model.model is not None}")
    print(f"   - Soil types count: {len(f2_model.get_soil_types())}")
    print(f"   - Crop types count: {len(f2_model.get_crop_types())}")
except Exception as e:
    print(f"❌ F2 Model failed: {e}")
    f2_model = None

try:
    print("\n🔄 Loading Price Model...")
    from models.price_model import PriceModel
    price_model = PriceModel()
    print(f"✅ Price Model loaded - Model: {price_model.model is not None}")
    print(f"   - Regions count: {len(price_model.get_available_regions())}")
    print(f"   - Commodities count: {len(price_model.get_available_commodities())}")
except Exception as e:
    print(f"❌ Price Model failed: {e}")
    price_model = None

try:
    print("\n🔄 Loading Crop Model...")
    from models.crop_model import CropModel
    crop_model = CropModel()
    print(f"✅ Crop Model loaded - Model: {crop_model.model is not None}")
    print(f"   - Available crops: {len(crop_model.get_available_crops())}")
    print(f"   - Scaler: {crop_model.scaler is not None}")
    print(f"   - LabelEncoder: {crop_model.le is not None}")
except Exception as e:
    print(f"❌ Crop Model failed: {e}")
    crop_model = None

# === NEW: Load Personal Assistant ===
try:
    print("\n🔄 Loading Personal Assistant...")
    from models.assistant import PersonalAssistant
    assistant = PersonalAssistant()
    print("✅ Personal Assistant loaded")
except Exception as e:
    print(f"❌ Personal Assistant failed: {e}")
    assistant = None

print("\n" + "=" * 50)
print("🎉 Model initialization complete!")
print(f"F2 Model: {'✅ READY' if f2_model and f2_model.model is not None else '❌ NOT READY'}")
print(f"Price Model: {'✅ READY' if price_model and price_model.model is not None else '❌ NOT READY'}")
print(f"Crop Model: {'✅ READY' if crop_model and crop_model.model is not None else '❌ NOT READY'}")
print(f"Assistant: {'✅ READY' if assistant else '❌ NOT READY'}")
print("=" * 50)

@app.route('/')
def home():
    print("\n🌐 Home page requested")
    return render_template('index.html',
                         f2_model=f2_model,
                         price_model=price_model,
                         crop_model=crop_model)

@app.route('/fertilizer')
def fertilizer_home():
    print(f"\n🌱 Fertilizer page requested")
    if f2_model is not None:
        soil_types = f2_model.get_soil_types()
        crop_types = f2_model.get_crop_types()
    else:
        soil_types = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red']
        crop_types = ['rice', 'wheat', 'maize', 'sugarcane', 'cotton']
    return render_template('fertilizer.html', 
                          soil_types=soil_types, 
                          crop_types=crop_types)

@app.route('/price')
def price_home():
    print(f"\n💰 Price page requested")
    if price_model is not None:
        regions = price_model.get_available_regions()
        commodities = price_model.get_available_commodities()
    else:
        regions = ['North', 'South', 'East', 'West']
        commodities = ['Tomato', 'Potato', 'Onion', 'Carrot']
    return render_template('price.html', 
                          regions=regions, 
                          commodities=commodities)

@app.route('/crop')
def crop_home():
    print(f"\n🌽 Crop page requested")
    if crop_model is not None:
        available_crops = crop_model.get_available_crops()
        feature_ranges = crop_model.get_feature_ranges()
        features = crop_model.FEATURES
    else:
        available_crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane']
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        feature_ranges = {
            'N': {'min': 0, 'max': 140, 'mean': 70},
            'P': {'min': 5, 'max': 145, 'mean': 50},
            'K': {'min': 5, 'max': 205, 'mean': 50},
            'temperature': {'min': 8, 'max': 44, 'mean': 25},
            'humidity': {'min': 14, 'max': 99, 'mean': 65},
            'ph': {'min': 3.5, 'max': 9.9, 'mean': 6.5},
            'rainfall': {'min': 20, 'max': 298, 'mean': 100}
        }
    return render_template('crop.html', 
                          available_crops=available_crops,
                          feature_ranges=feature_ranges,
                          features=features)

# === NEW: Assistant Route ===
@app.route('/ask-assistant', methods=['POST'])
def ask_assistant():
    """Handle chat messages from the personal assistant"""
    if assistant is None:
        return jsonify({'error': True, 'message': 'Assistant not available'}), 500
    
    user_message = request.form.get('message', '').strip()
    if not user_message:
        return jsonify({'error': True, 'message': 'No message provided'})
    
    response_text = assistant.respond(user_message)
    
    return jsonify({
        'error': False,
        'response': response_text
    })

@app.route('/predict-fertilizer', methods=['POST'])
def predict_fertilizer():
    if f2_model is None:
        return jsonify({'error': True, 'message': 'Fertilizer model not available'}), 500
    input_data = {
        'nitrogen': request.form.get('nitrogen'),
        'phosphorous': request.form.get('phosphorous'),
        'potassium': request.form.get('potassium'),
        'temperature': request.form.get('temperature'),
        'humidity': request.form.get('humidity'),
        'moisture': request.form.get('moisture'),
        'soil_type': request.form.get('soil_type'),
        'crop_type': request.form.get('crop_type')
    }
    result = f2_model.predict(input_data)
    return jsonify(result)

@app.route('/predict-price', methods=['POST'])
def predict_price():
    if price_model is None:
        return jsonify({'error': True, 'message': 'Price model not available'}), 500
    region = request.form.get('region')
    commodity = request.form.get('commodity')
    target_date = request.form.get('date')
    temp = request.form.get('temperature')
    rain = request.form.get('rainfall')
    result = price_model.predict_price(region, commodity, target_date, temp, rain)
    return jsonify(result)

@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    if crop_model is None:
        return jsonify({'error': True, 'message': 'Crop recommendation model not available'}), 500
    input_data = {
        'N': request.form.get('N'),
        'P': request.form.get('P'),
        'K': request.form.get('K'),
        'temperature': request.form.get('temperature'),
        'humidity': request.form.get('humidity'),
        'ph': request.form.get('ph'),
        'rainfall': request.form.get('rainfall')
    }
    result = crop_model.predict(input_data)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
