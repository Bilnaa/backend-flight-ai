"""
Routes pour le Machine Learning (fit et predict)
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas.ml import FitRequest, FitResponse, PredictRequest, PredictResponse
from app.services.ray_service import RayModelService

router = APIRouter()

# Instance globale du service Ray
_ray_service: Optional[RayModelService] = None


def get_ray_service() -> RayModelService:
    """Obtenir l'instance du service Ray (singleton)"""
    global _ray_service  # noqa: PLW0603
    if _ray_service is None:
        _ray_service = RayModelService()
    return _ray_service


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
