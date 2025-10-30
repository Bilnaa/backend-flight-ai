from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Airline, Airport

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):  # noqa: B008
    """Vérification de santé de l'API, de Ray et de la base de données"""
    try:
        # Vérifier que Ray fonctionne
        import ray

        ray_status = "healthy" if ray.is_initialized() else "not_initialized"

        # Vérifier la base de données
        try:
            airlines_count = db.query(Airline).count()
            airports_count = db.query(Airport).count()
            db_status = "healthy"


        except Exception as db_error:
            db_status = f"error: {str(db_error)}"
            airlines_count = 0
            airports_count = 0

        return {
            "status": "healthy" if ray_status == "healthy" and db_status == "healthy" else "degraded",
            "ray_status": ray_status,
            "database": {"status": db_status, "airlines_count": airlines_count, "airports_count": airports_count},
            "endpoints": ["/fit", "/predict", "/health", "/airlines", "/airports"]
        }

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service non disponible: {str(e)}") from e
