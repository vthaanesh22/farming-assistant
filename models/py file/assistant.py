# models/assistant.py
"""
HarvestMate Personal Assistant - Multilingual (English, Tamil, Telugu, Malayalam)
"""

import random
from models.f2_model import F2Model
from models.crop_model import CropModel
from models.price_model import PriceModel

class PersonalAssistant:
    def __init__(self):
        self.f2_model = None
        self.crop_model = None
        self.price_model = None

        print("🤖 Initializing Multilingual HarvestMate Personal Assistant...")

        try:
            self.f2_model = F2Model()
            print("✅ F2 model loaded in assistant")
        except Exception as e:
            print(f"⚠️ F2 model not available: {e}")

        try:
            self.crop_model = CropModel()
            print("✅ Crop model loaded in assistant")
        except Exception as e:
            print(f"⚠️ Crop model not available: {e}")

        try:
            self.price_model = PriceModel()
            print("✅ Price model loaded in assistant")
        except Exception as e:
            print(f"⚠️ Price model not available: {e}")

        print("🎉 Multilingual Personal Assistant ready!\n")

    def detect_language(self, text: str):
        """Simple language detection based on common words"""
        text_lower = text.lower()

        # Tamil indicators
        if any(word in text_lower for word in ['வணக்கம்', 'எப்படி', 'என்ன', 'உரம்', 'பயிர்', 'விலை']):
            return 'ta'

        # Telugu indicators
        if any(word in text_lower for word in ['నమస్కారం', 'ఎలా', 'ఏమి', 'ఎరువు', 'పంట', 'ధర']):
            return 'te'

        # Malayalam indicators
        if any(word in text_lower for word in ['നമസ്കാരം', 'എങ്ങനെ', 'എന്ത്', 'വളം', 'വിള', 'വില']):
            return 'ml'

        # Default to English
        return 'en'

    def respond(self, user_message: str):
        if not user_message or not isinstance(user_message, str):
            return "Sorry, I didn't understand that."

        lang = self.detect_language(user_message)
        msg = user_message.lower().strip()

        # ==================== ENGLISH ====================
        if lang == 'en':
            if any(g in msg for g in ['hi', 'hello', 'namaste', 'hey']):
                return "Namaste! 👋 I'm your HarvestMate AI Assistant. How can I help you today?"

            elif any(g in msg for g in ['fertilizer', 'urea', 'dap', 'potassium']):
                return "🌱 I can recommend the best fertilizer! Please go to the **Fertilizer Recommendation** page and enter your soil & crop details."

            elif any(g in msg for g in ['crop', 'which crop', 'recommend crop']):
                return "🌾 I can suggest the best crop for your land! Go to **Crop Recommendation** and enter N-P-K, temperature, humidity, pH & rainfall."

            elif any(g in msg for g in ['price', 'market price', 'tomato price']):
                return "💰 I can predict vegetable market prices! Go to **Market Price Predictor** and select region, commodity & date."

            elif any(g in msg for g in ['help', 'tip', 'advice']):
                tips = [
                    "Test your soil before applying fertilizer.",
                    "Crop rotation improves soil health.",
                    "Monitor weather regularly for better planning."
                ]
                return random.choice(tips)

            else:
                return "I'm here to help with crops, fertilizer, and market prices 🌾 What would you like to know?"

        # ==================== TAMIL ====================
        elif lang == 'ta':
            if any(g in msg for g in ['வணக்கம்', 'ஹலோ']):
                return "வணக்கம்! 👋 நான் உங்கள் HarvestMate உதவியாளர். இன்று எப்படி உதவட்டும்?"

            elif any(g in msg for g in ['உரம்', 'யூரியா', 'டிஏபி']):
                return "🌱 சிறந்த உரத்தை பரிந்துரைக்க முடியும்! **Fertilizer Recommendation** பக்கத்திற்கு சென்று மண் மற்றும் பயிர் விவரங்களை உள்ளிடவும்."

            elif any(g in msg for g in ['பயிர்', 'எந்த பயிர்']):
                return "🌾 உங்கள் நிலத்திற்கு சிறந்த பயிரை பரிந்துரைக்க முடியும்! **Crop Recommendation** பக்கத்தில் N-P-K, வெப்பநிலை, ஈரப்பதம், pH மற்றும் மழைப்பொழிவை உள்ளிடவும்."

            elif any(g in msg for g in ['விலை', 'சந்தை விலை']):
                return "💰 காய்கறி சந்தை விலையை கணிக்க முடியும்! **Market Price Predictor** பக்கத்தில் பகுதி, பொருள் மற்றும் தேதியை தேர்வு செய்யவும்."

            else:
                return "பயிர், உரம் அல்லது சந்தை விலை பற்றி உதவ வேண்டுமா? என்ன தெரிய வேண்டும்?"

        # ==================== TELUGU ====================
        elif lang == 'te':
            if any(g in msg for g in ['నమస్కారం', 'హలో']):
                return "నమస్కారం! 👋 నేను మీ HarvestMate సహాయకుడిని. ఈరోజు ఎలా సహాయపడాలి?"

            elif any(g in msg for g in ['ఎరువు', 'యూరియా']):
                return "🌱 ఉత్తమ ఎరువును సూచించగలను! **Fertilizer Recommendation** పేజీకి వెళ్లి మట్టి & పంట వివరాలు నమోదు చేయండి."

            elif any(g in msg for g in ['పంట', 'ఏ పంట']):
                return "🌾 మీ భూమికి ఉత్తమ పంటను సూచించగలను! **Crop Recommendation** లో N-P-K, ఉష్ణోగ్రత, తేమ, pH & వర్షపాతం నమోదు చేయండి."

            elif any(g in msg for g in ['ధర', 'మార్కెట్ ధర']):
                return "💰 కూరగాయల మార్కెట్ ధరలను అంచనా వేయగలను! **Market Price Predictor** లో ప్రాంతం, ఉత్పత్తి & తేదీ ఎంచుకోండి."

            else:
                return "పంట, ఎరువు లేదా మార్కెట్ ధర గురించి ఏమైనా అడగాలా?"

        # ==================== MALAYALAM ====================
        elif lang == 'ml':
            if any(g in msg for g in ['നമസ്കാരം', 'ഹലോ']):
                return "നമസ്കാരം! 👋 ഞാൻ നിങ്ങളുടെ HarvestMate അസിസ്റ്റന്റാണ്. ഇന്ന് എങ്ങനെ സഹായിക്കട്ടെ?"

            elif any(g in msg for g in ['വളം', 'യൂറിയ']):
                return "🌱 മികച്ച വളം ശുപാർശ ചെയ്യാം! **Fertilizer Recommendation** പേജിലേക്ക് പോയി മണ്ണിന്റെയും വിളയുടെയും വിശദാംശങ്ങൾ നൽകുക."

            elif any(g in msg for g in ['വിള', 'ഏത് വിള']):
                return "🌾 നിങ്ങളുടെ ഭൂമിക്ക് ഏറ്റവും അനുയോജ്യമായ വിള ശുപാർശ ചെയ്യാം! **Crop Recommendation** പേജിൽ N-P-K, താപനില, ഈർപ്പം, pH, മഴ എന്നിവ നൽകുക."

            elif any(g in msg for g in ['വില', 'മാർക്കറ്റ് വില']):
                return "💰 പച്ചക്കറി മാർക്കറ്റ് വില പ്രവചിക്കാം! **Market Price Predictor** പേജിൽ പ്രദേശം, ഉൽപ്പന്നം, തീയതി തിരഞ്ഞെടുക്കുക."

            else:
                return "വിള, വളം അല്ലെങ്കിൽ മാർക്കറ്റ് വിലയെ കുറിച്ച് സഹായം വേണോ?"

        # Fallback
        return "I'm here to help with crops, fertilizer, and prices 🌾 What do you need?"


# Quick test when running directly
if __name__ == "__main__":
    assistant = PersonalAssistant()
    print("\n--- Multilingual Assistant Test ---")
    tests = [
        "Hello, how are you?",
        "வணக்கம், உரம் பற்றி சொல்லுங்கள்",
        "నమస్కారం, ఏ పంట మంచిది?",
        "നമസ്കാരം, വിള വില എത്രയാണ്?",
        "Give me a farming tip"
    ]
    for t in tests:
        print(f"User: {t}")
        print(f"Assistant: {assistant.respond(t)}\n")