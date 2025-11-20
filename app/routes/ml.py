"""
Routes pour le Machine Learning (fit et predict)
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas.ml import (
    FitRequest,
    FitResponse,
    PredictBookingRequest,
    PredictBookingResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.booking_model_service import BookingModelService
from app.services.ray_service import RayModelService

router = APIRouter()

# Instances globales des services
_ray_service: Optional[RayModelService] = None
_booking_model_service: Optional[BookingModelService] = None


def get_ray_service() -> RayModelService:
    """Obtenir l'instance du service Ray (singleton)"""
    global _ray_service  # noqa: PLW0603
    if _ray_service is None:
        _ray_service = RayModelService()
    return _ray_service


def get_booking_model_service() -> BookingModelService:
    """Obtenir l'instance du service de modèle de réservation (singleton)"""
    global _booking_model_service  # noqa: PLW0603
    if _booking_model_service is None:
        try:
            _booking_model_service = BookingModelService(models_dir="models")
        except FileNotFoundError:
            # Les modèles ne sont pas encore créés, ce qui est normal avant le premier 'fit'
            _booking_model_service = BookingModelService(models_dir="models")
            # Ne pas appeler _load_model_and_stats ici, car les fichiers n'existent pas
    return _booking_model_service


@router.post("/fit", response_model=FitResponse)
async def fit_model(request: FitRequest):
    """
    Entraîner le modèle avec Ray

    Le modèle peut être pré-entraîné dans ai-models/main.ipynb
    Cet endpoint permet de ré-entraîner avec de nouvelles données
    """
    try:
        if len(request.months) != len(request.delays):
            raise HTTPException(status_code=400, detail="Les listes 'months' et 'delays' doivent avoir la même taille")

        if len(request.months) == 0:
            raise HTTPException(status_code=400, detail="Les données d'entraînement ne peuvent pas être vides")

        # Préparer les données pour Ray
        X = np.array(request.months).reshape(-1, 1)
        y = np.array(request.delays)

        # Entraîner avec Ray
        ray_service = get_ray_service()
        result_message = await ray_service.fit(X, y)

        return FitResponse(
            message=result_message, samples_count=len(request.months), model_path="models/trained_model.pkl"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement: {str(e)}") from e


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Faire une prédiction avec Ray

    Utilise le modèle entraîné (via /fit ou le .pkl)
    le .pkl devrait être dans le dossier models/trained_model.pkl
    """
    try:
        if not (1 <= request.month <= 12):
            raise HTTPException(status_code=400, detail="Le mois doit être entre 1 et 12")
        if not (1 <= request.day <= 31):
            raise HTTPException(status_code=400, detail="Le jour doit être entre 1 et 31")
        if not request.origin_airport or not request.dest_airport:
            raise HTTPException(status_code=400, detail="Les aéroports de départ et de destination sont requis")

        # Prédiction avec Ray
        ray_service = get_ray_service()
        predicted_delay, using_model = await ray_service.predict(
            month=request.month,
            day=request.day,
            origin_airport=request.origin_airport,
            dest_airport=request.dest_airport,
        )

        return PredictResponse(
            day=request.day,
            month=request.month,
            origin_airport=request.origin_airport,
            dest_airport=request.dest_airport,
            predicted_delay=predicted_delay,
            model_version="v1.0",
            using_model=using_model,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}") from e


@router.post("/fit-booking-model", response_model=FitResponse)
async def fit_booking_model():
    """
    Entraîner le modèle de prédiction au moment de la réservation.

    Ce processus recalcule les statistiques historiques et ré-entraîne le modèle
    Random Forest, puis sauvegarde les fichiers .pkl nécessaires.
    """
    try:
        service = get_booking_model_service()
        service.fit(data_path="data/flights.csv")
        return FitResponse(
            message="Modèle de réservation entraîné avec succès.",
            samples_count=10000,  # Taille de l'échantillon utilisé dans le service
            model_path=service.model_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement du modèle de réservation: {str(e)}")


@router.post("/predict-booking", response_model=PredictBookingResponse)
async def predict_booking(request: PredictBookingRequest):
    """
    Prédiction au moment de la réservation avec le nouveau modèle (ai-models)

    Utilise le modèle entraîné dans ai-models (Random Forest avec statistiques historiques)
    Fichiers requis : models/modele_best.pkl et models/historical_stats.pkl
    """
    try:
        # Validation basique
        if not (1 <= request.MONTH <= 12):
            raise HTTPException(status_code=400, detail="Le mois doit être entre 1 et 12")
        if not (1 <= request.DAY_OF_WEEK <= 7):
            raise HTTPException(status_code=400, detail="Le jour de la semaine doit être entre 1 et 7")
        if not (0 <= request.SCHEDULED_DEPARTURE <= 2359):
            raise HTTPException(status_code=400, detail="L'heure de départ doit être entre 0000 et 2359")
        if not request.AIRLINE or not request.ORIGIN_AIRPORT or not request.DESTINATION_AIRPORT:
            raise HTTPException(status_code=400, detail="La compagnie et les aéroports sont requis")

        # Obtenir le service
        service = get_booking_model_service()

        # Préparer les données pour le modèle
        booking_data = {
            "AIRLINE": request.AIRLINE,
            "ORIGIN_AIRPORT": request.ORIGIN_AIRPORT,
            "DESTINATION_AIRPORT": request.DESTINATION_AIRPORT,
            "MONTH": request.MONTH,
            "DAY_OF_WEEK": request.DAY_OF_WEEK,
            "SCHEDULED_DEPARTURE": request.SCHEDULED_DEPARTURE,
            "DISTANCE": request.DISTANCE,
        }

        # Faire la prédiction
        predicted_delay = service.predict(booking_data)

        return PredictBookingResponse(
            predicted_delay=float(predicted_delay),
            model_version="v3.0",  # Mise à jour de la version
            using_model=True,
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Le modèle de prédiction n'est pas disponible. "
            "Veuillez l'entraîner d'abord via l'endpoint /fit-booking-model.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction booking: {str(e)}") from e
