"""
Service pour le modèle de prédiction au moment de la réservation.

Ce service gère :
- Le chargement du modèle et des statistiques historiques.
- L'entraînement (fit) du modèle depuis les données brutes.
- La prédiction (predict) pour une nouvelle réservation.
"""

import os
import pickle
from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class BookingModelService:
    """Service pour le modèle de prédiction de retard au moment de la réservation."""

    def __init__(self, models_dir: str = "models"):
        """
        Initialise le service en chargeant le modèle et les statistiques.

        Args:
            models_dir: Le répertoire où les modèles sont stockés.
        """
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "modele_best.pkl")
        self.stats_path = os.path.join(models_dir, "historical_stats.pkl")

        self._model = None
        self._encoders = None
        self._feature_names = None
        self._stats = None

        self._load_model_and_stats()

    def _load_model_and_stats(self):
        """Charge le modèle et les statistiques depuis les fichiers pickle."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.stats_path):
            raise FileNotFoundError(
                "Les fichiers 'modele_best.pkl' ou 'historical_stats.pkl' sont manquants. "
                "Veuillez d'abord entraîner le modèle via l'endpoint /fit-booking-model."
            )

        with open(self.model_path, "rb") as f:
            model_package = pickle.load(f)
            self._model = model_package["model"]
            self._encoders = model_package["encoders"]
            self._feature_names = model_package["feature_names"]

        with open(self.stats_path, "rb") as f:
            self._stats = pickle.load(f)

    def predict(self, booking_data: Dict[str, Any]) -> float:
        """
        Prédit le retard de vol pour une réservation donnée.

        Args:
            booking_data: Dictionnaire contenant les informations de la réservation.

        Returns:
            Le retard prédit en minutes.
        """
        if self._model is None or self._stats is None:
            self._load_model_and_stats()

        # Créer un DataFrame à partir des données de réservation
        df = pd.DataFrame([booking_data])

        # Fusionner avec les statistiques historiques
        df = self._merge_historical_stats(df)

        # Appliquer l'ingénierie des caractéristiques
        df = self._apply_feature_engineering(df)

        # Encoder les caractéristiques catégorielles
        for col, encoder in self._encoders.items():
            if col in df.columns:
                # Gérer les valeurs inconnues
                known_values = encoder.classes_
                df[col] = df[col].apply(lambda x: x if x in known_values else "unknown")
                # Transformer les données
                df[col] = encoder.transform(df[col])

        # S'assurer que toutes les colonnes du modèle sont présentes
        for col in self._feature_names:
            if col not in df.columns:
                df[col] = 0  # Remplir avec une valeur par défaut (ex: 0)

        # Réorganiser les colonnes pour correspondre à l'entraînement
        df = df[self._feature_names]

        # Faire la prédiction
        prediction = self._model.predict(df)
        return float(prediction[0])

    def fit(self, data_path: str = "data/flights.csv"):
        """
        Entraîne le modèle de bout en bout et sauvegarde les artefacts.

        Args:
            data_path: Chemin vers le fichier flights.csv.
        """
        # 1. Charger les données
        raw_df = pd.read_csv(data_path, low_memory=False)
        df = raw_df[raw_df.ARRIVAL_DELAY.notna()].copy()

        # 2. Calculer et sauvegarder les statistiques historiques
        self._calculate_and_save_stats(df)

        # 3. Préparer les données pour le modèle
        cols = [
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "MONTH",
            "DAY_OF_WEEK",
            "SCHEDULED_DEPARTURE",
            "DISTANCE",
            "ARRIVAL_DELAY",
        ]
        df_model = df[cols].copy()

        # Fusionner les statistiques
        df_model = self._merge_historical_stats(df_model)

        # 4. Ingénierie des caractéristiques
        df_model = self._apply_feature_engineering(df_model)

        # Remplir les NaNs restants
        for col in [
            "hist_avg_delay",
            "hist_std_delay",
            "airline_avg_delay",
            "route_avg_delay",
            "origin_avg_delay",
            "dest_avg_delay",
        ]:
            if col in df_model.columns:
                df_model[col] = df_model[col].fillna(df_model["ARRIVAL_DELAY"].mean())
        if "hist_count" in df_model.columns:
            df_model["hist_count"] = df_model["hist_count"].fillna(0)

        # 5. Échantillonner et préparer les données d'entraînement
        sample_size = min(10000, len(df_model))
        df_sample = df_model.sample(n=sample_size, random_state=42)

        X = df_sample.drop(["ARRIVAL_DELAY", "SCHEDULED_DEPARTURE"], axis=1, errors="ignore")
        y = df_sample["ARRIVAL_DELAY"]

        # Encoder les caractéristiques catégorielles
        categorical_features = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
        encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            # Gérer les valeurs inconnues en ajoutant une classe 'unknown'
            all_values = pd.concat([X[col], pd.Series(["unknown"])]).unique()
            le.fit(all_values.astype(str))
            X[col] = le.transform(X[col].astype(str))
            encoders[col] = le

        # 6. Entraîner le modèle
        # Utilisation de RandomForest comme dans le notebook
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features=0.5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        # 7. Sauvegarder le modèle et les encodeurs
        model_package = {
            "model": model,
            "encoders": encoders,
            "feature_names": X.columns.tolist(),
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(model_package, f)

        # Recharger le modèle et les stats pour la suite
        self._load_model_and_stats()

    def _calculate_and_save_stats(self, df: pd.DataFrame):
        """Calcule et sauvegarde les statistiques historiques."""
        route_airline_stats = (
            df.groupby(["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH"])["ARRIVAL_DELAY"]
            .agg([("hist_avg_delay", "mean"), ("hist_std_delay", "std"), ("hist_count", "count")])
            .reset_index()
        )

        airline_stats = df.groupby("AIRLINE")["ARRIVAL_DELAY"].agg([("airline_avg_delay", "mean")]).reset_index()

        route_stats = (
            df.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH"])["ARRIVAL_DELAY"]
            .agg([("route_avg_delay", "mean")])
            .reset_index()
        )

        origin_stats = (
            df.groupby(["ORIGIN_AIRPORT", "MONTH"])["ARRIVAL_DELAY"].agg([("origin_avg_delay", "mean")]).reset_index()
        )

        dest_stats = (
            df.groupby(["DESTINATION_AIRPORT", "MONTH"])["ARRIVAL_DELAY"]
            .agg([("dest_avg_delay", "mean")])
            .reset_index()
        )

        stats_package = {
            "route_airline_stats": route_airline_stats,
            "airline_stats": airline_stats,
            "route_stats": route_stats,
            "origin_stats": origin_stats,
            "dest_stats": dest_stats,
        }

        with open(self.stats_path, "wb") as f:
            pickle.dump(stats_package, f)

        self._stats = stats_package

    def _merge_historical_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fusionne le DataFrame avec les statistiques historiques."""
        df = df.merge(
            self._stats["route_airline_stats"],
            on=["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH"],
            how="left",
        )
        df = df.merge(self._stats["airline_stats"], on="AIRLINE", how="left")
        df = df.merge(self._stats["route_stats"], on=["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH"], how="left")
        df = df.merge(self._stats["origin_stats"], on=["ORIGIN_AIRPORT", "MONTH"], how="left")
        df = df.merge(self._stats["dest_stats"], on=["DESTINATION_AIRPORT", "MONTH"], how="left")
        return df

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique l'ingénierie des caractéristiques au DataFrame."""
        if "SCHEDULED_DEPARTURE" in df.columns:
            df["DEPARTURE_HOUR"] = df["SCHEDULED_DEPARTURE"] // 100
            df["IS_MORNING"] = (df["DEPARTURE_HOUR"] < 12).astype(int)
            df["IS_EVENING"] = (df["DEPARTURE_HOUR"] >= 18).astype(int)

        if "DAY_OF_WEEK" in df.columns:
            df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 6).astype(int)

        if "MONTH" in df.columns:
            df["IS_SUMMER"] = df["MONTH"].isin([6, 7, 8]).astype(int)
            df["IS_HOLIDAY"] = df["MONTH"].isin([11, 12]).astype(int)

        if "DISTANCE" in df.columns:
            df["IS_SHORT"] = (df["DISTANCE"] < 500).astype(int)
            df["IS_LONG"] = (df["DISTANCE"] > 2000).astype(int)

        return df
