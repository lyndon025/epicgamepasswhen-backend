import pickle
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from datetime import datetime
import os

class GameServicePredictor:
    def __init__(
        self,
        csv_path,
        xgb_model_path,
        publisher_stats_path,
        publisher_encoder_path,
        platform_name,
        avg_repeat_interval,
        repeat_confidence_mult,
        date_column,
        date_format,
        model_quality_mult=1.0,
        max_confidence_cap=95,
        disclaimer="",
        platform_check=None,
    ):
        self.platform_name = platform_name
        self.avg_repeat_interval = avg_repeat_interval
        self.repeat_confidence_mult = repeat_confidence_mult
        self.model_quality_mult = model_quality_mult
        self.max_confidence_cap = max_confidence_cap
        self.disclaimer = disclaimer
        self.platform_check = platform_check

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["game_name"].notna()].copy()

        # Parse dates from the ACTUAL column name using robust parser
        self.df["added_to_service"] = self.df[date_column].apply(self._parse_date_robust)
        self.df["release_date"] = self.df["release_date"].apply(self._parse_date_robust)

        print(f"✓ Loaded {len(self.df)} games from {csv_path}")
        print(f"✓ Date column: {date_column}")
        print(
            f"✓ Valid 'added_to_service' dates: {self.df['added_to_service'].notna().sum()}"
        )

        with open(xgb_model_path, "rb") as f:
            self.xgb_model = pickle.load(f)
        with open(publisher_encoder_path, "rb") as f:
            self.publisher_encoder = pickle.load(f)

        self.publisher_stats = pd.read_csv(publisher_stats_path)
        self.median_metacritic = 75

    def _parse_date_robust(self, date_str):
        if pd.isna(date_str):
            return pd.NaT
        
        date_str = str(date_str).strip().strip('"') # Remove quotes if present
        
        formats = [
            "%m/%d/%Y",      # 01/20/2026
            "%Y-%m-%d",      # 2026-01-20
            "%B %d, %Y",     # July 17, 2025
            "%b %d, %Y",     # Jul 17, 2025
            "%d-%b-%y",      # 20-Jan-26 (rare but possible)
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
                
        # Fallback
        return pd.to_datetime(date_str, errors="coerce")

    def normalize_title(self, title):
        """Remove all punctuation except spaces, convert to lowercase"""
        return "".join(c for c in title.lower() if c.isalnum() or c.isspace())

    def extract_numeral(self, title):
        """Extract numeral suffix (Roman or digit) from title"""
        # Strip all punctuation before searching for numerals
        cleaned_title = re.sub(r"[^a-z0-9ivxlcdm\s]", "", title.lower())
        match = re.search(r"\b([ivxlcdm]+|\d+)$", cleaned_title)
        return match.group(1) if match else None

    def _check_first_party_publisher(self, publisher):
        """Check if publisher is a first-party publisher for this platform"""
        if not publisher:
            return None

        publisher_lower = publisher.lower()

        if self.platform_name == "Xbox Game Pass":
            # Core Microsoft Studios
            ms_keywords = ["microsoft", "xbox game studios", "xbox publishing"]

            # Microsoft-owned studios (Activision-Blizzard & Bethesda acquisitions)
            activision_keywords = ["activision", "blizzard"]
            bethesda_keywords = ["bethesda", "zenimax"]

            # Check if it's core Microsoft
            if any(keyword in publisher_lower for keyword in ms_keywords):
                return {
                    "tier": "First-Party Publisher",
                    "category": "Day One (Xbox Game Pass Ultimate)",
                    "confidence": 99,
                    "reasoning": f"Microsoft first-party title. Available Day One on Xbox Game Pass Ultimate & PC Game Pass (Standard tier may not include Day One titles).",
                    "first_party": True,
                    "available_on": ["Xbox Game Pass Ultimate", "PC Game Pass"],
                    "predicted_months": 0.0,
                    "predicted_days": 0.0,
                    "publisher_game_count": None,
                    "publisher_consistency": None,
                    "sample_size": None,
                }

            # Check if it's Activision-Blizzard (Microsoft-owned since 2023)
            if any(keyword in publisher_lower for keyword in activision_keywords):
                return {
                    "tier": "Microsoft-Owned (Activision-Blizzard)",
                    "category": "Very Likely (Staggered Release)",
                    "confidence": 85,
                    "reasoning": f"{publisher} is owned by Microsoft. Games are joining Game Pass on a staggered schedule. Note: New releases require Xbox Game Pass Ultimate or PC Game Pass for Day One access.",
                    "first_party": True,
                    "available_on": ["Xbox Game Pass Ultimate", "PC Game Pass"],
                    "predicted_months": 3.0,
                    "predicted_days": 90.0,
                    "publisher_game_count": None,
                    "publisher_consistency": None,
                    "sample_size": None,
                }

            # Check if it's Bethesda (Microsoft-owned since 2021)
            if any(keyword in publisher_lower for keyword in bethesda_keywords):
                return {
                    "tier": "Microsoft-Owned (Bethesda/ZeniMax)",
                    "category": "Very Likely (Within 6 Months)",
                    "confidence": 90,
                    "reasoning": f"{publisher} is owned by Microsoft. Most titles join Game Pass Day One (requires Ultimate or PC Game Pass).",
                    "first_party": True,
                    "available_on": ["Xbox Game Pass Ultimate", "PC Game Pass"],
                    "predicted_months": 3.0,
                    "predicted_days": 90.0,
                    "publisher_game_count": None,
                    "publisher_consistency": None,
                    "sample_size": None,
                }

        elif self.platform_name == "PS Plus Extra":
            sony_keywords = ["sony", "playstation studios", "sie", "sony interactive"]
            if any(keyword in publisher_lower for keyword in sony_keywords):
                return {
                    "tier": "First-Party Publisher",
                    "category": "Likely (within 12-24 months)",
                    "confidence": 75,
                    "reasoning": f"Sony first-party title from {publisher}. PlayStation Studios games typically join PS Plus Extra catalog within 12-24 months.",
                    "first_party": True,
                    "predicted_months": 18.0,
                    "predicted_days": 540.0,
                    "publisher_game_count": None,
                    "publisher_consistency": None,
                    "sample_size": None,
                }

        return None

    def _calculate_confidence(
        self,
        sample_size,
        variance_coefficient=None,
        has_metacritic=False,
        is_repeat=False,
    ):
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
            return "within 6 months"
        elif months <= 12:
            return "within 6-12 months"
        elif months <= 24:
            return "more than 12 months"
        elif months <= 48:
            return "more than 24 months"
        else:
            return "as good as never (many years)"

    def check_if_appeared(self, game_name):
        """Check if game has appeared in service before - with fuzzy matching"""

        # First try exact match
        appearances = self.df[self.df["game_name"].str.lower() == game_name.lower()]

        # If no exact match, try fuzzy matching
        if len(appearances) == 0:
            best_match = None
            best_score = 0
            threshold = 0.90  # Threshold for fuzzy matching

            # Extract numeral from query
            query_numeral = self.extract_numeral(game_name)
            normalized_query = self.normalize_title(game_name)

            for csv_game_name in self.df["game_name"].unique():
                normalized_csv = self.normalize_title(csv_game_name)
                similarity = SequenceMatcher(
                    None, normalized_query, normalized_csv
                ).ratio()

                csv_numeral = self.extract_numeral(csv_game_name)

                # Only accept match if numerals match or both None
                if (
                    similarity > best_score
                    and similarity >= threshold
                    and query_numeral == csv_numeral
                ):
                    best_score = similarity
                    best_match = csv_game_name

            if best_match:
                print(
                    f"  Fuzzy match: '{game_name}' ≈ '{best_match}' ({best_score:.0%})"
                )
                appearances = self.df[
                    self.df["game_name"].str.lower() == best_match.lower()
                ]
            else:
                print(f"  No match found for '{game_name}'")
                return None
        else:
            print(f"  Exact match: '{game_name}'")

        if len(appearances) == 0:
            return None

        # Parse dates from the parsed 'added_to_service' column
        dates = appearances["added_to_service"].dropna().sort_values()
        if len(dates) == 0:
            return {
                "appeared": True,
                "repeat_count": len(appearances),
                "last_appearance": None,
            }

        result = {
            "appeared": True,
            "repeat_count": len(dates),
            "last_appearance": dates.iloc[-1],
        }

        if len(dates) >= 2:
            intervals = [
                (dates.iloc[i + 1] - dates.iloc[i]).days for i in range(len(dates) - 1)
            ]
            result["avg_interval_months"] = np.mean(intervals) / 30
            result["cv"] = (
                np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
            )

        return result

    def predict_repeat(self, game_name):
        """TIER 1: Check if game appeared before - MOST RELIABLE"""
        try:
            history = self.check_if_appeared(game_name)
            if not history or not history.get("appeared"):
                print(f"  {game_name}: Not in history")
                return None

            last_appearance = history.get("last_appearance")
            if last_appearance is None or pd.isna(last_appearance):
                print(f"  {game_name}: In history but no date")
                return None

            months_since = (datetime.now() - last_appearance).days / 30
            print(f"  {game_name}: Last appeared {months_since:.1f} months ago")

            # --- HUMBLE BUNDLE SPECIAL LOGIC ---
            if self.platform_name == "Humble Choice":
                # Calculate theoretical wait time if it WERE to repeat
                if history["repeat_count"] == 1:
                    theoretical_months = max(0, self.avg_repeat_interval - months_since)
                else:
                    avg_interval = history.get("avg_interval_months", self.avg_repeat_interval)
                    theoretical_months = max(0, avg_interval - months_since)

                last_date_str = last_appearance.strftime("%B %Y")

                return {
                    "category": "Very unlikely (Already Appeared)",
                    "confidence": 95,
                    "predicted_months": 0,
                    "reasoning": f"This game has already appeared in a Humble Choice/Monthly bundle ({last_date_str}). Repeat appearances have never happened before (as of January 2026).",
                    "sample_size": history["repeat_count"],
                    "tier": "Historical Lookup (Humble No-Repeat Rule)",
                    "recently_appeared": False,
                    "months_since_last": 0,
                    "theoretical_wait_time": theoretical_months, # For technical display
                    "last_appearance_date": last_date_str
                }
            # -----------------------------------

            last_appearance = history.get("last_appearance")
            if last_appearance is None or pd.isna(last_appearance):
                print(f"  {game_name}: In history but no date")
                return None

            months_since = (datetime.now() - last_appearance).days / 30
            print(f"  {game_name}: Last appeared {months_since:.1f} months ago")

            if history["repeat_count"] == 1:
                predicted_months = max(0, self.avg_repeat_interval - months_since)
                confidence = self._calculate_confidence(1, None, False, True)
                reasoning = f"Appeared once {months_since:.1f} months ago on {self.platform_name}. Avg repeat interval: ~{self.avg_repeat_interval:.0f} months."
            else:
                avg_interval = history.get(
                    "avg_interval_months", self.avg_repeat_interval
                )
                predicted_months = max(0, avg_interval - months_since)
                confidence = self._calculate_confidence(
                    history["repeat_count"], history.get("cv"), False, True
                )
                reasoning = f"Appeared {history['repeat_count']} times. Avg interval: {avg_interval:.0f} months. Last: {months_since:.0f} months ago."

            # Check if game appeared recently (within 12 months)
            recently_appeared = months_since <= 12

            if self.disclaimer:
                reasoning += f" {self.disclaimer}"

            return {
                "category": self._months_to_bucket(predicted_months),
                "confidence": confidence,
                "predicted_months": float(predicted_months),
                "reasoning": reasoning,
                "sample_size": history["repeat_count"],
                "tier": "Historical Lookup (Repeat Pattern)",
                "recently_appeared": recently_appeared,
                "months_since_last": float(months_since),
            }
        except Exception as e:
            print(f"Error in predict_repeat: {e}")
            import traceback

            traceback.print_exc()
            return None

    def predict_new_xgb(
        self, game_name, publisher, metacritic_score=None, release_date=None
    ):
        """TIER 2: Predict new game using XGBoost"""
        if publisher not in self.publisher_encoder.classes_:
            return {
                "category": "unknown (no record of publisher in service)",
                "confidence": 0,
                "reasoning": f"Publisher '{publisher}' not found in {self.platform_name} training data.",
                "tier": "Unknown",
            }

        pub_stats = self.publisher_stats[self.publisher_stats["publisher"] == publisher]
        if len(pub_stats) == 0:
            return {
                "category": "unknown (no record of publisher in service)",
                "confidence": 0,
                "reasoning": f"No statistics for publisher '{publisher}' on {self.platform_name}.",
                "tier": "Unknown",
            }

        pub_stats = pub_stats.iloc[0]
        meta_score = metacritic_score if metacritic_score else self.median_metacritic
        publisher_encoded = self.publisher_encoder.transform([publisher])[0]

        features = np.array(
            [
                [
                    meta_score,
                    publisher_encoded,
                    pub_stats["pub_avg_days"],
                    pub_stats["pub_count"],
                    pub_stats["pub_cv"],
                ]
            ]
        )

        predicted_log_days = self.xgb_model.predict(features)[0]

        # INVERT LOG TRANSFORMATION
        # All models (Xbox, PS, Epic, Humble) are trained with log(days), so we must invert it.
        predicted_days = np.exp(predicted_log_days)

        predicted_months = predicted_days / 30

        confidence = self._calculate_confidence(
            int(pub_stats["pub_count"]),
            pub_stats["pub_cv"],
            metacritic_score is not None,
            False,
        )

        category = self._months_to_bucket(predicted_months)
        reasoning = f"XGBoost prediction for {self.platform_name}: {predicted_days:.0f} days ({predicted_months:.0f} months). Publisher '{publisher}' has {int(pub_stats['pub_count'])} games on service."

        if self.disclaimer:
            reasoning += f" {self.disclaimer}"

        return {
            "category": category,
            "confidence": confidence,
            "predicted_months": float(predicted_months),
            "predicted_days": float(predicted_days),
            "reasoning": reasoning,
            "publisher_game_count": int(pub_stats["pub_count"]),
            "publisher_consistency": float(pub_stats["pub_cv"]),
            "tier": "XGBoost ML Prediction (New Game)",
        }

    def predict(
        self,
        game_name,
        publisher=None,
        metacritic_score=None,
        platforms=None,
        release_date=None,
    ):
        """Main prediction method - Priority checks"""

        # PRIORITY 1: First-party publisher check
        if publisher:
            first_party_result = self._check_first_party_publisher(publisher)
            if first_party_result:
                # For Xbox first-party, check if game is already released
                if self.platform_name == "Xbox Game Pass" and release_date:
                    try:
                        release_dt = pd.to_datetime(release_date, errors="coerce")
                        if pd.notna(release_dt):
                            now = datetime.now()

                            if release_dt > now:
                                # Game hasn't released yet - Day One
                                days_until_release = (release_dt - now).days
                                first_party_result["category"] = (
                                    "Day One (Available at Game Release)"
                                )
                                first_party_result["reasoning"] = (
                                    f"Microsoft first-party title. Will be available Day One on Xbox Game Pass when it releases in {days_until_release} days."
                                )
                            else:
                                # Game already released - should be available now
                                days_since_release = (now - release_dt).days
                                first_party_result["category"] = (
                                    "Available Now (Very Soon if Not Yet Added)"
                                )
                                first_party_result["reasoning"] = (
                                    f"Microsoft first-party title released {days_since_release} days ago. Should already be available on Xbox Game Pass Ultimate and PC Game Pass, or coming very soon."
                                )
                    except Exception as e:
                        print(f"Error parsing release date: {e}")

                return {
                    "game_name": game_name,
                    "publisher": publisher,
                    **first_party_result,
                }

        # PRIORITY 2: Platform compatibility check (OPTIONAL)
        if self.platform_check and platforms:
            try:
                platform_result = self.platform_check(platforms, self.platform_name)
                if platform_result:
                    return {"game_name": game_name, **platform_result}
            except Exception as e:
                print(f"Platform check failed: {e}")

        # PRIORITY 3: Check for repeat pattern (old games)
        try:
            repeat_pred = self.predict_repeat(game_name)
            if repeat_pred:
                print(f"✓ Using repeat pattern for {game_name}")
                return {"game_name": game_name, **repeat_pred}
        except Exception as e:
            print(f"Repeat prediction error: {e}")

        # PRIORITY 4: XGBoost prediction for NEW games
        if not publisher:
            return {
                "game_name": game_name,
                "tier": "Unknown",
                "category": "unknown (no record of publisher in service)",
                "confidence": 0,
                "reasoning": "No publisher provided and no historical data available.",
            }

        print(f"→ {game_name} not in history, using ML prediction")
        new_pred = self.predict_new_xgb(
            game_name, publisher, metacritic_score, release_date
        )
        return {"game_name": game_name, "publisher": publisher, **new_pred}
