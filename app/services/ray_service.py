"""
Service Ray pour l'orchestration du modèle

L'orchestration Ray fonctionne ainsi :
1. Ray Actors (@ray.remote class) : Crée une instance du modèle qui reste en mémoire
   sur un worker dédié. Le modèle est chargé une fois et réutilisé pour toutes les
   prédictions.

2. Ray Tasks (.remote()) : Quand on appelle predictor.predict.remote(), Ray envoie
   cette tâche au worker qui héberge l'acteur. Plusieurs prédictions peuvent être
   traitées en parallèle.

3. Ray.get() : Récupère les résultats de manière asynchrone. Si plusieurs prédictions
   sont en cours, elles peuvent être récupérées ensemble.

Avantages de cette architecture :
- Le modèle reste chargé en mémoire (pas de rechargement à chaque requête)
- Les prédictions peuvent être distribuées sur plusieurs workers
- Scalabilité horizontale facile
- Performance optimale pour les prédictions en production
"""

import os
import pickle
from typing import Dict, Optional

import numpy as np
import ray

# Initialiser Ray si ce n'est pas déjà fait
if not ray.is_initialized():
    ray_address = os.getenv("RAY_ADDRESS", "auto")
    # Traiter "auto" ou vide comme mode local
    if ray_address and ray_address.strip().lower() not in ("", "auto"):
        try:
            # Tente de se connecter à un cluster Ray existant (ex: ray://ray:10001)
            print(f"Tentative de connexion au cluster Ray à l'adresse: {ray_address}")
            ray.init(address=ray_address, ignore_reinit_error=True)
            print(f"Connecté au cluster Ray: {ray_address}")
        except Exception as e:
            print(f"Échec de la connexion au cluster Ray: {e}")
            # Si la connexion échoue, bascule en mode local
            if not ray.is_initialized():
                print("Bascule en mode Ray local")
                ray.init(ignore_reinit_error=True)
    else:
        # Mode local si RAY_ADDRESS n'est pas défini ou est "auto"
        if not ray.is_initialized():
            print("Démarrage de Ray en mode local")
            ray.init(ignore_reinit_error=True)


@ray.remote
class ModelPredictor:
    """
    Acteur Ray pour les prédictions de modèle

    Un acteur Ray est une instance persistante qui maintient l'état en mémoire.
    Ici, le modèle est chargé une fois et réutilisé pour toutes les prédictions.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialiser le modèle - chargé une seule fois au démarrage"""
        # Utiliser os.getenv() directement car on ne peut pas importer app.config dans le conteneur Ray
        self.model_path = model_path or os.getenv("MODEL_PATH", "models/modele.pkl")
        self.model = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Charger le modèle pré-entraîné depuis le disque"""
        try:
            # Afficher des informations de débogage
            abs_path = os.path.abspath(self.model_path)
            current_dir = os.getcwd()
            print("[ModelPredictor] Tentative de chargement du modèle:")
            print(f"  Chemin relatif: {self.model_path}")
            print(f"  Chemin absolu: {abs_path}")
            print(f"  Répertoire courant: {current_dir}")
            print(f"  Le fichier existe: {os.path.exists(self.model_path)}")

            # Essayer aussi avec le chemin absolu
            paths_to_try = [self.model_path, abs_path]
            if not os.path.isabs(self.model_path):
                # Essayer aussi avec /app/models/ si on est dans un conteneur
                container_path = f"/app/{self.model_path}"
                paths_to_try.append(container_path)

            model_file = None
            for path in paths_to_try:
                if os.path.exists(path):
                    model_file = path
                    break

            if model_file:
                print(f"[ModelPredictor] Chargement du modèle depuis: {model_file}")
                with open(model_file, "rb") as f:
                    self.model = pickle.load(f)
                self.model_loaded = True
                print(f"[ModelPredictor] ✓ Modèle chargé avec succès depuis {model_file}")
            else:
                print("[ModelPredictor] ✗ Modèle non trouvé. Chemins testés:")
                for path in paths_to_try:
                    print(f"  - {path} (existe: {os.path.exists(path)})")
                print("  Placez le modèle entraîné dans l'un de ces emplacements")
                self.model_loaded = False
        except (FileNotFoundError, pickle.UnpicklingError, OSError) as e:
            print(f"[ModelPredictor] ✗ Erreur lors du chargement du modèle: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Entraîner le modèle avec de nouvelles données

        Args:
            X: Matrice des features (ex: mois)
            y: Valeurs cibles (ex: retards)

        Returns:
            Message de confirmation
        """
        try:
            from sklearn.linear_model import LinearRegression

            # Créer et entraîner le modèle
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.model_loaded = True

            # Sauvegarder le modèle
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)

            return f"Modèle entraîné avec {len(X)} échantillons et sauvegardé dans {self.model_path}"

        except Exception as e:
            return f"Erreur lors de l'entraînement: {e}"

    def predict(
        self,
        month: int,
        day: Optional[int] = None,
        origin_airport: Optional[str] = None,
        dest_airport: Optional[str] = None,
        additional_features: Optional[Dict] = None,
    ) -> tuple[float, bool]:
        """
        Faire une prédiction avec le modèle pré-entraîné

        Args:
            month: Mois (1-12)
            additional_features: Features additionnelles optionnelles (non utilisé pour l'instant)

        Returns:
            Tuple (valeur prédite du retard, using_model: True si modèle utilisé, False si fallback)
        """
        # Ignorer additional_features pour l'instant (pourra être utilisé plus tard)
        del additional_features  # Supprimer l'avertissement de paramètre non utilisé

        if self.model_loaded and self.model is not None:
            # Utiliser le modèle pré-entraîné. Si le modèle n'attend qu'une feature,
            # on n'utilise que le mois pour rester compatible avec l'entraînement actuel.
            try:
                n_features = getattr(self.model, "n_features_in_", 1)
            except Exception:
                n_features = 1

            if n_features == 1:
                X = np.array([[month]])
            else:
                # Tentative de vectorisation simple si le modèle accepte plus de features
                safe_day = (day if day is not None else 15)
                # Encodage déterministe très simple pour les aéroports (hash mod 10)
                oa = (abs(hash(origin_airport)) % 10) if origin_airport else 0
                da = (abs(hash(dest_airport)) % 10) if dest_airport else 0
                X = np.array([[month, safe_day, oa, da]])

            prediction = self.model.predict(X)[0]
            return (float(prediction), True)  # Modèle utilisé
        else:
            # Fallback vers logique basique si modèle non disponible
            base_delays = {
                1: 10.5,
                2: 12.3,
                3: 15.2,
                4: 18.1,
                5: 20.5,
                6: 25.3,
                7: 28.7,
                8: 26.2,
                9: 22.1,
                10: 18.5,
                11: 14.2,
                12: 11.8,
            }
            base = float(base_delays.get(month, 15.0))

            # Ajustements heuristiques simples
            safe_day = day if isinstance(day, int) and 1 <= day <= 31 else 15
            # Légère hausse en fin de mois
            day_adj = 0.1 * max(0, safe_day - 20)

            # Ajustement par aéroports (ex: hubs = +2, autres = +0.5 cumulable)
            busy_hubs = {"JFK", "LAX", "SFO", "ORD", "ATL", "DFW", "CDG", "LHR"}
            origin_adj = (
                2.0
                if (origin_airport and origin_airport.upper() in busy_hubs)
                else (0.5 if origin_airport else 0.0)
            )
            dest_adj = 2.0 if dest_airport and dest_airport.upper() in busy_hubs else 0.5 if dest_airport else 0.0

            estimate = base + day_adj + origin_adj + dest_adj
            return (float(max(0.0, estimate)), False)  # Fallback utilisé


class RayModelService:
    """
    Service pour utiliser Ray avec le modèle pré-entraîné

    Ce service orchestre les prédictions via Ray :
    - Crée un acteur qui maintient le modèle en mémoire
    - Distribue les prédictions sur les workers disponibles
    - Gère le chargement et le rechargement du modèle
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialiser le service Ray avec le modèle pré-entraîné"""
        # Importer settings ici (pas au niveau du module) car on est dans le conteneur API
        # L'acteur ModelPredictor ne peut pas importer app.config car il s'exécute dans Ray
        try:
            from app.config import settings
            default_path = settings.MODEL_PATH
        except ImportError:
            # Fallback si settings n'est pas disponible
            default_path = os.getenv("MODEL_PATH", "models/modele.pkl")

        # Utiliser le chemin depuis les settings si non fourni
        self.model_path = model_path or default_path
        # Créer un acteur Ray qui maintient le modèle en mémoire
        # L'acteur est créé une seule fois et réutilisé pour toutes les prédictions
        # .remote() est ajouté dynamiquement par le décorateur @ray.remote
        self.predictor = ModelPredictor.remote(self.model_path)  # type: ignore[attr-defined]

    async def fit(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Entraîner le modèle en utilisant Ray

        Args:
            X: Matrice des features
            y: Valeurs cibles

        Returns:
            Message de confirmation de l'entraînement
        """
        # Utiliser Ray pour l'entraînement distribuée
        result = ray.get(self.predictor.fit.remote(X, y))
        return result

    async def predict(
        self,
        month: int,
        day: Optional[int] = None,
        origin_airport: Optional[str] = None,
        dest_airport: Optional[str] = None,
        additional_data: Optional[Dict] = None,
    ) -> tuple[float, bool]:
        """
        Faire une prédiction en utilisant Ray

        Comment ça marche :
        1. .remote() envoie la tâche de prédiction à l'acteur Ray
        2. L'acteur exécute la prédiction avec le modèle en mémoire
        3. ray.get() récupère le résultat de manière asynchrone

        Args:
            month: Mois pour la prédiction
            day: Jour du mois de départ (1-31)
            origin_airport: Aéroport de départ (code IATA)
            dest_airport: Aéroport de destination (code IATA)
            additional_data: Données additionnelles

        Returns:
            Tuple (valeur prédite, using_model: True si modèle utilisé, False si fallback)
        """
        # Utiliser Ray pour la prédiction distribuée
        # .remote() envoie la tâche à l'acteur, ray.get() récupère le résultat
        prediction, using_model = ray.get(
            self.predictor.predict.remote(
                month,
                day,
                origin_airport,
                dest_airport,
                additional_data,
            )
        )

        return (prediction, using_model)

    def shutdown(self):
        """Arrêter Ray proprement"""
        if ray.is_initialized():
            ray.shutdown()
