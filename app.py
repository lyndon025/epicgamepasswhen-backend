from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
CORS(app, origins=["*"])

port = int(os.environ.get('PORT', 5000))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ============================================================================
# UNIVERSAL PREDICTOR CLASS
# ============================================================================

class GameServicePredictor:
    def __init__(self, csv_path, xgb_model_path, publisher_stats_path, publisher_encoder_path, 
                 platform_name, avg_repeat_interval, repeat_confidence_mult, date_column, date_format,
                 model_quality_mult=1.0, max_confidence_cap=95, disclaimer="", platform_check=None):
        self.platform_name = platform_name
        self.avg_repeat_interval = avg_repeat_interval
        self.repeat_confidence_mult = repeat_confidence_mult
        self.model_quality_mult = model_quality_mult
        self.max_confidence_cap = max_confidence_cap
        self.disclaimer = disclaimer
        self.platform_check = platform_check
        
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['game_name'].notna()].copy()
        self.df['added_to_service'] = pd.to_datetime(self.df[date_column], format=date_format, errors='coerce')
        self.df['release_date'] = pd.to_datetime(self.df['release_date'], format=date_format, errors='coerce')
        
        with open(xgb_model_path, 'rb') as f:
            self.xgb_model = pickle.load(f)
        with open(publisher_encoder_path, 'rb') as f:
            self.publisher_encoder = pickle.load(f)
        
        self.publisher_stats = pd.read_csv(publisher_stats_path)
        self.median_metacritic = 75
    
    def _check_first_party_publisher(self, publisher):
        """Check if publisher is a first-party publisher for this platform"""
        if not publisher:
            return None
        
        publisher_lower = publisher.lower()
        
        # Xbox Game Pass - Microsoft first-party
        if self.platform_name == 'Xbox Game Pass':
            ms_keywords = ['microsoft', 'xbox game studios', 'xbox publishing']
            if any(keyword in publisher_lower for keyword in ms_keywords):
                return {
                    'tier': 'First-Party Publisher',
                    'category': 'Day One (Game Pass Ultimate & PC)',
                    'confidence': 99,
                    'reasoning': f"Microsoft first-party title. All {publisher} games release Day One on Xbox Game Pass Ultimate and PC Game Pass.",
                    'first_party': True,
                    'available_on': ['Xbox Game Pass Ultimate', 'PC Game Pass'],
                    # ADD REAL DATA FIELDS
                    'predicted_months': 0,  # Day One means 0 months wait
                    'predicted_days': 0,  # Day One means 0 days wait
                    'publisher_game_count': None,  # Not applicable for first-party
                    'publisher_consistency': None,  # Not applicable for first-party
                    'sample_size': None  # Not applicable for first-party
                }
        
        # PS Plus - Sony first-party
        elif self.platform_name == 'PS Plus Extra':
            sony_keywords = ['sony', 'playstation studios', 'sie', 'sony interactive']
            if any(keyword in publisher_lower for keyword in sony_keywords):
                return {
                    'tier': 'First-Party Publisher',
                    'category': 'Likely (within 12-24 months)',
                    'confidence': 75,
                    'reasoning': f"Sony first-party title from {publisher}. PlayStation Studios games typically join PS Plus Extra catalog within 12-24 months, though exact timing varies by strategic decisions.",
                    'first_party': True,
                    # ADD REAL DATA FIELDS
                    'predicted_months': 18,  # Average 18 months for Sony first-party
                    'predicted_days': 540,  # 18 months = 540 days
                    'publisher_game_count': None,  # Not applicable for first-party
                    'publisher_consistency': None,  # Not applicable for first-party
                    'sample_size': None  # Not applicable for first-party
                }
        
        return None

    
    def _calculate_confidence(self, sample_size, variance_coefficient=None, has_metacritic=False, is_repeat=False):
        if is_repeat:
            if sample_size >= 3:
                base = 85
            elif sample_size == 2:
                base = 75
            else:
                base = 65
            base = int(base * self.repeat_confidence_mult)
        else:
            if sample_size >= 20:
                base = 80
            elif sample_size >= 10:
                base = 70
            elif sample_size >= 5:
                base = 60
            elif sample_size >= 3:
                base = 50
            else:
                base = 40
        
        if variance_coefficient is not None:
            if variance_coefficient < 0.3:
                base += 10
            elif variance_coefficient < 0.5:
                base += 5
            elif variance_coefficient > 0.8:
                base -= 10
        
        if has_metacritic:
            base += 5
        
        base = int(base * self.model_quality_mult)
        return max(min(int(base), self.max_confidence_cap), 5)
    
    def _months_to_bucket(self, months):
        if months <= 6:
            return 'within 6 months'
        elif months <= 12:
            return 'within 6-12 months'
        elif months <= 24:
            return 'more than 12 months'
        elif months <= 48:
            return 'more than 24 months'
        else:
            return 'as good as never (many years)'
    
    def check_if_appeared(self, game_name):
        appearances = self.df[self.df['game_name'].str.lower() == game_name.lower()]
        if len(appearances) == 0:
            return None
        
        dates = appearances['added_to_service'].dropna().sort_values()
        if len(dates) == 0:
            return {'appeared': True, 'repeat_count': len(appearances)}
        
        result = {
            'appeared': True,
            'repeat_count': len(dates),
            'last_appearance': dates.iloc[-1]
        }
        
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
            predicted_months = max(0, self.avg_repeat_interval - months_since)
            confidence = self._calculate_confidence(1, None, False, True)
            reasoning = f"Appeared once {months_since:.1f} months ago on {self.platform_name}. Avg repeat: ~{self.avg_repeat_interval:.0f} months."
        else:
            avg_interval = history['avg_interval_months']
            predicted_months = max(0, avg_interval - months_since)
            confidence = self._calculate_confidence(history['repeat_count'], history['cv'], False, True)
            reasoning = f"Appeared {history['repeat_count']} times on {self.platform_name}. Avg interval: {avg_interval:.0f} months. {months_since:.1f} months since last."
        
        if self.disclaimer:
            reasoning += f" Note: {self.disclaimer}"
        
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
                'reasoning': f"Publisher '{publisher}' not found in {self.platform_name} training data.",
                'tier': 'Unknown'
            }
        
        pub_stats = self.publisher_stats[self.publisher_stats['publisher'] == publisher]
        if len(pub_stats) == 0:
            return {
                'category': 'unknown (no record of publisher in service)',
                'confidence': 0,
                'reasoning': f"No statistics for publisher '{publisher}' on {self.platform_name}.",
                'tier': 'Unknown'
            }
        
        pub_stats = pub_stats.iloc[0]
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
        reasoning = f"XGBoost prediction for {self.platform_name}: {predicted_days:.0f} days ({predicted_months:.0f} months). Publisher '{publisher}' has {int(pub_stats['pub_count'])} games on service."
        
        if self.disclaimer:
            reasoning += f" Note: {self.disclaimer}"
        
        return {
            'category': category,
            'confidence': confidence,
            'predicted_months': predicted_months,
            'predicted_days': predicted_days,
            'reasoning': reasoning,
            'publisher_game_count': int(pub_stats['pub_count']),
            'publisher_consistency': float(pub_stats['pub_cv']),
            'tier': 'XGBoost ML Prediction (New Game)'
        }
    
    def predict(self, game_name, publisher=None, metacritic_score=None, platforms=None):
        # PRIORITY 1: First-party publisher check (NEW)
        if publisher:
            first_party_result = self._check_first_party_publisher(publisher)
            if first_party_result:
                return {'game_name': game_name, 'publisher': publisher, **first_party_result}
        
        # PRIORITY 2: Platform compatibility check
        if self.platform_check and platforms:
            platform_result = self.platform_check(platforms, self.platform_name)
            if platform_result:
                return {'game_name': game_name, **platform_result}
        
        # TIER 1: Check for repeat pattern
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
                'reasoning': 'No publisher provided and no historical data available.'
            }
        
        new_pred = self.predict_new_xgb(game_name, publisher, metacritic_score)
        return {'game_name': game_name, 'publisher': publisher, **new_pred}

# ============================================================================
# PLATFORM CHECKS
# ============================================================================

def check_pc_platform(platforms_data, platform_name):
    if not platforms_data:
        return None
    pc_keywords = ['pc', 'windows', 'linux', 'macos']
    platform_names = [p.get('platform', {}).get('name', '').lower() for p in platforms_data]
    is_pc = any(keyword in ' '.join(platform_names) for keyword in pc_keywords)
    if not is_pc:
        return {
            'tier': 'Platform Check',
            'category': 'not on pc (console exclusive)',
            'confidence': 95,
            'reasoning': f"Game is not available on PC. Available on: {', '.join([p for p in platform_names if p])}. Epic Games Store only offers PC games.",
            'platforms': platform_names
        }
    return None

def check_xbox_platform(platforms_data, platform_name):
    if not platforms_data:
        return None
    xbox_keywords = ['xbox', 'pc', 'windows']
    platform_names = [p.get('platform', {}).get('name', '').lower() for p in platforms_data]
    is_xbox = any(keyword in ' '.join(platform_names) for keyword in xbox_keywords)
    if not is_xbox:
        return {
            'tier': 'Platform Check',
            'category': 'not on xbox/pc',
            'confidence': 95,
            'reasoning': f"Game is not available on Xbox or PC. Available on: {', '.join([p for p in platform_names if p])}. Xbox Game Pass requires Xbox or PC platform.",
            'platforms': platform_names
        }
    return None

def check_playstation_platform(platforms_data, platform_name):
    if not platforms_data:
        return None
    ps_keywords = ['playstation', 'ps4', 'ps5']
    platform_names = [p.get('platform', {}).get('name', '').lower() for p in platforms_data]
    is_ps = any(keyword in ' '.join(platform_names) for keyword in ps_keywords)
    if not is_ps:
        return {
            'tier': 'Platform Check',
            'category': 'not on playstation',
            'confidence': 95,
            'reasoning': f"Game is not available on PlayStation. Available on: {', '.join([p for p in platform_names if p])}. PS Plus requires PlayStation platform.",
            'platforms': platform_names
        }
    return None

# ============================================================================
# INITIALIZE PREDICTORS
# ============================================================================

epic_predictor = GameServicePredictor(
    csv_path=os.path.join(BASE_DIR, 'Epic.csv'),
    xgb_model_path=os.path.join(MODEL_DIR, 'xgb_epic_model.pkl'),
    publisher_stats_path=os.path.join(MODEL_DIR, 'publisher_statistics.csv'),
    publisher_encoder_path=os.path.join(MODEL_DIR, 'publisher_encoder.pkl'),
    platform_name='Epic Games',
    avg_repeat_interval=18.9,
    repeat_confidence_mult=1.0,
    date_column='Added to Service',
    date_format='%m/%d/%Y',
    model_quality_mult=1.0,
    max_confidence_cap=95,
    disclaimer="",
    platform_check=check_pc_platform
)

xbox_predictor = GameServicePredictor(
    csv_path=os.path.join(BASE_DIR, 'Xbox.csv'),
    xgb_model_path=os.path.join(MODEL_DIR, 'xgb_xbox_model.pkl'),
    publisher_stats_path=os.path.join(MODEL_DIR, 'publisher_statistics_xbox.csv'),
    publisher_encoder_path=os.path.join(MODEL_DIR, 'publisher_encoder_xbox.pkl'),
    platform_name='Xbox Game Pass',
    avg_repeat_interval=24.0,
    repeat_confidence_mult=0.75,
    date_column='Added to Service',
    date_format='%Y-%m-%d',
    model_quality_mult=0.75,
    max_confidence_cap=80,
    disclaimer="Moderate uncertainty - Game Pass patterns vary",
    platform_check=check_xbox_platform
)

# PS Plus Predictor
psplus_predictor = GameServicePredictor(
    csv_path=os.path.join(BASE_DIR, 'PS.csv'),
    xgb_model_path=os.path.join(MODEL_DIR, 'xgb_psplus_model.pkl'),
    publisher_stats_path=os.path.join(MODEL_DIR, 'publisher_statistics_psplus.csv'),
    publisher_encoder_path=os.path.join(MODEL_DIR, 'publisher_encoder_psplus.pkl'),
    platform_name='PS Plus Extra',
    avg_repeat_interval=24.0,
    repeat_confidence_mult=0.75,
    date_column='Added to Service',
    date_format='%Y-%m-%d',
    model_quality_mult=0.6,
    max_confidence_cap=70,
    disclaimer="High uncertainty - PS Plus catalog patterns are unpredictable",
    platform_check=check_playstation_platform
)


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    platform = data.get('platform', 'epic')
    
    if platform == 'epic':
        predictor = epic_predictor
    elif platform == 'gamepass':
        predictor = xbox_predictor
    elif platform == 'psplus':  # ADD THIS LINE
        predictor = psplus_predictor  # ADD THIS LINE
    else:
        return jsonify({'error': f'Unknown platform: {platform}'}), 400

    
    result = predictor.predict(
        game_name=data.get('game_name'),
        publisher=data.get('publisher'),
        metacritic_score=data.get('metacritic_score'),
        platforms=data.get('platforms')
    )
    
    # FIXED: Better serialization that handles lists and arrays
    serializable = {}
    for key, value in result.items():
        # Handle lists and arrays directly
        if isinstance(value, (list, tuple)):
            serializable[key] = value
        # Handle numpy numbers
        elif isinstance(value, (np.integer, np.floating)):
            if np.isnan(value):
                serializable[key] = None
            else:
                serializable[key] = float(value)
        # Handle pandas NA/NaN (scalars only)
        elif value is None or (isinstance(value, float) and pd.isna(value)):
            serializable[key] = None
        # Everything else (strings, bools, etc.)
        else:
            serializable[key] = value
    
    return jsonify(serializable)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models': {
            'epic': 'v4.2 XGBoost + Two-Tier',
            'gamepass': 'v1.0 XGBoost + Two-Tier + First-Party',
            'psplus': 'v1.0 XGBoost + Two-Tier + First-Party'  # ADD THIS LINE
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
