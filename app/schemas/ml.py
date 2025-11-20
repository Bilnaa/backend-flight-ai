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


class PredictBookingRequest(BaseModel):
    """Requête pour la prédiction au moment de la réservation (nouveau modèle)"""

    AIRLINE: str
    ORIGIN_AIRPORT: str
    DESTINATION_AIRPORT: str
    MONTH: int
    DAY_OF_WEEK: int
    SCHEDULED_DEPARTURE: int
    DISTANCE: int


class PredictBookingResponse(BaseModel):
    """Réponse de prédiction au moment de la réservation"""

    predicted_delay: float
    model_version: str = "v2.0"
    using_model: bool = True


class ExplainRequest(BaseModel):
    """Requête pour l'explication d'une prédiction (SHAP)"""

    AIRLINE: str
    ORIGIN_AIRPORT: str
    DESTINATION_AIRPORT: str
    MONTH: int
    DAY_OF_WEEK: int
    SCHEDULED_DEPARTURE: int
    DISTANCE: int


class ExplainResponse(BaseModel):
    """Réponse d'explication avec le graphique SHAP"""

    image_base64: str
    model_version: str = "v3.0"
