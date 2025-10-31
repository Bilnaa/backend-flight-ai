from pydantic import BaseModel


class FitRequest(BaseModel):
    """Requête pour l'entraînement du modèle"""

    months: list[int]
    delays: list[float]


class FitResponse(BaseModel):
    """Réponse après l'entraînement"""

    message: str
    samples_count: int
    model_path: str


class PredictRequest(BaseModel):
    """Requête pour la prédiction"""

    day: int
    month: int
    origin_airport: str
    dest_airport: str


class PredictResponse(BaseModel):
    """Réponse de prédiction"""

    day: int
    month: int
    origin_airport: str
    dest_airport: str
    predicted_delay: float
    model_version: str
    using_model: bool  # True si le modèle ML est utilisé, False si fallback


