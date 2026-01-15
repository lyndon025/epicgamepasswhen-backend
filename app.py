from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys

# Add the current directory to the path so we can import services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.predictor import GameServicePredictor
from services.platform_checks import (
    check_pc_platform,
    check_xbox_platform,
    check_playstation_platform,
)

app = Flask(__name__)

# CORS Configuration - Allow specific origins
allowed_origins = [
    "https://epic-gamepass-when.vercel.app",  # Production frontend
    "https://epic-gamepass-when-git-main-lyndon025s-projects.vercel.app",
    "https://epic-gamepass-when-git-dev-lyndon025s-projects.vercel.app",
    "https://epic-gamepass-when.onrender.com",
    "http://localhost:5173",  # Local Vite dev server
    "http://localhost:5174",  # Alternate local port
    "http://localhost:3000",  # Local fallback
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:3000",
]

CORS(
    app,
    origins=allowed_origins,
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)

port = int(os.environ.get("PORT", 5000))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================================
# INITIALIZE PREDICTORS
# ============================================================================

epic_predictor = GameServicePredictor(
    csv_path=os.path.join(BASE_DIR, "Epic.csv"),
    xgb_model_path=os.path.join(MODEL_DIR, "xgb_epic_model.pkl"),
    publisher_stats_path=os.path.join(MODEL_DIR, "publisher_statistics.csv"),
    publisher_encoder_path=os.path.join(MODEL_DIR, "publisher_encoder.pkl"),
    platform_name="Epic Games",
    avg_repeat_interval=18.9,
    repeat_confidence_mult=1.0,
    date_column="Added to Service",
    date_format="%m/%d/%Y",
    model_quality_mult=1.0,
    max_confidence_cap=95,
    disclaimer="",
    platform_check=check_pc_platform,
)

xbox_predictor = GameServicePredictor(
    csv_path=os.path.join(BASE_DIR, "Xbox.csv"),
    xgb_model_path=os.path.join(MODEL_DIR, "xgb_xbox_model.pkl"),
    publisher_stats_path=os.path.join(MODEL_DIR, "publisher_statistics_xbox.csv"),
    publisher_encoder_path=os.path.join(MODEL_DIR, "publisher_encoder_xbox.pkl"),
    platform_name="Xbox Game Pass",
    avg_repeat_interval=24.0,
    repeat_confidence_mult=1.25,
    date_column="Added to Service",
    date_format="%m/%d/%Y",
    model_quality_mult=0.75,
    max_confidence_cap=90,
    disclaimer="Moderate uncertainty - Game Pass patterns vary",
    platform_check=check_xbox_platform,
)

psplus_predictor = GameServicePredictor(
    csv_path=os.path.join(BASE_DIR, "PS.csv"),
    xgb_model_path=os.path.join(MODEL_DIR, "xgb_psplus_model.pkl"),
    publisher_stats_path=os.path.join(MODEL_DIR, "publisher_statistics_psplus.csv"),
    publisher_encoder_path=os.path.join(MODEL_DIR, "publisher_encoder_psplus.pkl"),
    platform_name="PS Plus Extra",
    avg_repeat_interval=24.0,
    repeat_confidence_mult=1.25,
    date_column="Added to Service",
    date_format="%m/%d/%Y",
    model_quality_mult=0.6,
    max_confidence_cap=80,
    disclaimer="High uncertainty - PS Plus catalog patterns are unpredictable",
    platform_check=check_playstation_platform,
)


# ============================================================================
# API ROUTES
# ============================================================================



humble_predictor = GameServicePredictor(
    csv_path=os.path.join(BASE_DIR, "HB.csv"),
    xgb_model_path=os.path.join(MODEL_DIR, "xgb_humblebundle_model.pkl"),
    publisher_stats_path=os.path.join(MODEL_DIR, "publisher_statistics_humblebundle.csv"),
    publisher_encoder_path=os.path.join(MODEL_DIR, "publisher_encoder_humblebundle.pkl"),
    platform_name="Humble Choice",
    avg_repeat_interval=24.0,
    repeat_confidence_mult=1.0,
    date_column="Added to Service",
    date_format="%m/%d/%Y",
    model_quality_mult=0.7,
    max_confidence_cap=85,
    disclaimer="Prediction based on Humble Choice history",
    platform_check=check_pc_platform,
)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json

    # DEBUG: Print all data received
    print(f"\n{'='*70}")
    print("RECEIVED DATA:")
    print(f"  game_name: {data.get('game_name')}")
    print(f"  release_date: {data.get('release_date')}")
    print(f"  metacritic_score: {data.get('metacritic_score')}")
    print(f"{'='*70}\n")

    platform = data.get("platform", "epic")
    game_name = data.get("game_name", "Unknown")
    publisher = data.get("publisher")
    metacritic_score = data.get("metacritic_score")
    platforms = data.get("platforms")

    if platform == "epic":
        predictor = epic_predictor
    elif platform == "gamepass":
        predictor = xbox_predictor
    elif platform == "psplus":
        predictor = psplus_predictor
    elif platform == "humble":
        predictor = humble_predictor
    else:
        return jsonify({"error": f"Unknown platform: {platform}"}), 400

    try:
        if platforms and not isinstance(platforms, list):
            platforms = None

        if metacritic_score:
            try:
                metacritic_score = float(metacritic_score)
                if metacritic_score < 0 or metacritic_score > 100:
                    metacritic_score = None
            except (ValueError, TypeError):
                metacritic_score = None

        result = predictor.predict(
            game_name=game_name,
            publisher=publisher,
            metacritic_score=metacritic_score,
            platforms=platforms,
            release_date=data.get("release_date"),
        )

        def serialize_value(value):
            if value is None:
                return None
            elif isinstance(value, bool):
                return bool(value)
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (np.integer, np.floating)):
                try:
                    if np.isnan(value):
                        return None
                except (TypeError, ValueError):
                    pass
                return float(value)
            elif isinstance(value, (int, float)):
                try:
                    if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                        return None
                except (TypeError, ValueError):
                    pass
                return float(value) if isinstance(value, float) else int(value)
            else:
                return str(value)

        serializable = {}
        for key, value in result.items():
            try:
                serializable[key] = serialize_value(value)
            except Exception as e:
                print(f"Error serializing '{key}': {e}")
                serializable[key] = None

        return jsonify(serializable)

    except Exception as e:
        print(f"Error in predict(): {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()

        return (
            jsonify(
                {
                    "game_name": game_name,
                    "error": f"{type(e).__name__}: {str(e)}",
                    "category": "error",
                    "confidence": 0,
                    "tier": "Error",
                    "reasoning": "Backend error occurred",
                }
            ),
            500,
        )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "models": {
                "epic": "v4.2 XGBoost + Two-Tier",
                "gamepass": "v1.0 XGBoost + Two-Tier + First-Party",
                "psplus": "v1.0 XGBoost + Two-Tier",
                "humble": "v1.0 XGBoost + Two-Tier"
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
