from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os


app = Flask(__name__)
CORS(app, origins=["*"]) # Allow all origins for now
# Get port from environment variable for deployment
port = int(os.environ.get('PORT', 5000))

# Load models and artifacts (same as before)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

with open(os.path.join(MODEL_DIR, 'xgb_epic_model.pkl'), 'rb') as f:
    xgb_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'publisher_encoder.pkl'), 'rb') as f:
    publisher_encoder = pickle.load(f)

epic_df = pd.read_csv(os.path.join(BASE_DIR, 'Epic.csv'))
epic_df = epic_df[epic_df['game_name'].notna()].copy()
epic_df['added_to_service'] = pd.to_datetime(epic_df['Added to Service'], format='%m/%d/%Y', errors='coerce')
epic_df['release_date'] = pd.to_datetime(epic_df['release_date'], format='%m/%d/%Y', errors='coerce')

publisher_stats = pd.read_csv(os.path.join(MODEL_DIR, 'publisher_statistics.csv'))
median_metacritic = 75

# NEW: PC Platform Check Function
def is_on_pc(platforms_data):
    """
    Check if game is available on PC platforms
    platforms_data: list of platform dicts from RAWG API
    Returns: (is_pc, platform_names)
    """
    if not platforms_data:
        return None, []
    
    pc_keywords = ['pc', 'windows', 'linux', 'macos']
    platform_names = [p.get('platform', {}).get('name', '').lower() for p in platforms_data]
    
    is_pc = any(keyword in ' '.join(platform_names) for keyword in pc_keywords)
    return is_pc, platform_names

# Two-Tier Predictor Class (with PC check)
class EpicGamePredictorXGB:
    def __init__(self, epic_df, xgb_model, publisher_encoder, publisher_stats, median_meta):
        self.epic_df = epic_df
        self.xgb_model = xgb_model
        self.publisher_encoder = publisher_encoder
        self.publisher_stats = publisher_stats
        self.median_metacritic = median_meta
    
    def _calculate_confidence(self, sample_size, variance_coefficient=None, has_metacritic=False, is_repeat=False):
        if is_repeat:
            base = 85 if sample_size >= 3 else (75 if sample_size == 2 else 65)
        else:
            if sample_size >= 20: base = 80
            elif sample_size >= 10: base = 70
            elif sample_size >= 5: base = 60
            elif sample_size >= 3: base = 50
            else: base = 40
        
        if variance_coefficient is not None:
            if variance_coefficient < 0.3: base += 10
            elif variance_coefficient < 0.5: base += 5
            elif variance_coefficient > 0.8: base -= 10
        
        if has_metacritic: base += 5
        return max(min(int(base), 95), 5)
    
    def _months_to_bucket(self, months):
        if months <= 6: return 'within 6 months'
        elif months <= 12: return 'within 6-12 months'
        elif months <= 24: return 'more than 12 months'
        elif months <= 48: return 'more than 24 months'
        else: return 'as good as never (many years)'
    
    def check_if_appeared(self, game_name):
        appearances = self.epic_df[self.epic_df['game_name'].str.lower() == game_name.lower()]
        if len(appearances) == 0:
            return None
        
        dates = appearances['added_to_service'].dropna().sort_values()
        if len(dates) == 0:
            return {'appeared': True, 'repeat_count': len(appearances)}
        
        result = {'appeared': True, 'repeat_count': len(dates), 'last_appearance': dates.iloc[-1]}
        
        if len(dates) >= 2:
            intervals = [(dates.iloc[i+1] - dates.iloc[i]).days for i in range(len(dates)-1)]
            result['avg_interval_months'] = np.mean(intervals) / 30
            result['cv'] = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        return result
    
    def predict_repeat(self, game_name):
        history = self.check_if_appeared(game_name)
        if not history:
            return None
        
        months_since = (datetime.now() - history['last_appearance']).days / 30
        
        if history['repeat_count'] == 1:
            predicted_months = max(0, 18.9 - months_since)
            confidence = self._calculate_confidence(1, None, False, True)
            reasoning = f"Appeared once {months_since:.1f} months ago. Avg repeat: ~19 months."
        else:
            avg_interval = history['avg_interval_months']
            predicted_months = max(0, avg_interval - months_since)
            confidence = self._calculate_confidence(history['repeat_count'], history['cv'], False, True)
            reasoning = f"Appeared {history['repeat_count']} times. Avg: {avg_interval:.0f} months. {months_since:.1f} months since last."
        
        return {
            'category': self._months_to_bucket(predicted_months),
            'confidence': confidence,
            'predicted_months': predicted_months,
            'reasoning': reasoning,
            'sample_size': history['repeat_count'],
            'tier': 'Historical Lookup (Repeat Pattern)'
        }
    
    def predict_new_xgb(self, game_name, publisher, metacritic_score=None):
        if publisher not in self.publisher_encoder.classes_:
            return {
                'category': 'unknown (no record of publisher in service)',
                'confidence': 0,
                'reasoning': f"Publisher '{publisher}' not in training data.",
                'tier': 'Unknown'
            }
        
        pub_stats = self.publisher_stats[self.publisher_stats['publisher'] == publisher].iloc[0]
        
        meta_score = metacritic_score if metacritic_score else self.median_metacritic
        publisher_encoded = self.publisher_encoder.transform([publisher])[0]
        
        features = np.array([[
            meta_score,
            publisher_encoded,
            pub_stats['pub_avg_days'],
            pub_stats['pub_count'],
            pub_stats['pub_cv']
        ]])
        
        predicted_days = self.xgb_model.predict(features)[0]
        predicted_months = predicted_days / 30
        
        confidence = self._calculate_confidence(
            int(pub_stats['pub_count']),
            pub_stats['pub_cv'],
            metacritic_score is not None,
            False
        )
        
        category = self._months_to_bucket(predicted_months)
        reasoning = f"XGBoost prediction: {predicted_days:.0f} days ({predicted_months:.0f} months). Publisher '{publisher}' has {int(pub_stats['pub_count'])} games."
        
        return {
            'category': category,
            'confidence': confidence,
            'predicted_months': predicted_months,
            'reasoning': reasoning,
            'publisher_game_count': int(pub_stats['pub_count']),
            'publisher_consistency': float(pub_stats['pub_cv']),
            'tier': 'XGBoost ML Prediction (New Game)'
        }
    
    def predict(self, game_name, publisher=None, metacritic_score=None, platforms=None):
        # NEW: Check if game is on PC
        if platforms is not None:
            is_pc, platform_names = is_on_pc(platforms)
            
            if is_pc is False:
                return {
                    'game_name': game_name,
                    'tier': 'Platform Check',
                    'category': 'not on pc (console exclusive)',
                    'confidence': 95,
                    'reasoning': f"Game is not available on PC. Available on: {', '.join([p for p in platform_names if p])}. Epic Games Store only offers PC games.",
                    'platforms': platform_names
                }
        
        # TIER 1: Check repeat
        repeat_pred = self.predict_repeat(game_name)
        if repeat_pred:
            return {'game_name': game_name, **repeat_pred}
        
        # TIER 2: XGBoost prediction
        if not publisher:
            return {
                'game_name': game_name,
                'tier': 'Unknown',
                'category': 'unknown (no record of publisher in service)',
                'confidence': 0,
                'reasoning': 'No publisher provided.'
            }
        
        new_pred = self.predict_new_xgb(game_name, publisher, metacritic_score)
        return {'game_name': game_name, 'publisher': publisher, **new_pred}

# Initialize predictor
predictor = EpicGamePredictorXGB(epic_df, xgb_model, publisher_encoder, publisher_stats, median_metacritic)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    result = predictor.predict(
        game_name=data.get('game_name'),
        publisher=data.get('publisher'),
        metacritic_score=data.get('metacritic_score'),
        platforms=data.get('platforms')  # NEW: Pass platform data
    )
    
    # JSON serialization with NaN handling
    serializable = {}
    for key, value in result.items():
        if isinstance(value, (np.integer, np.floating)):
            if pd.isna(value):
                serializable[key] = None
            else:
                serializable[key] = float(value)
        elif pd.isna(value):
            serializable[key] = None
        else:
            serializable[key] = value
    
    return jsonify(serializable)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'Epic v4.2 XGBoost + Two-Tier + PC Check'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)