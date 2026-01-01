from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from crop_recommendation.db.database import SessionLocal
from crop_recommendation.db.models import Crop
from crop_recommendation.logic.npk_estimator import estimate_npk, initialize_npk_from_soil

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/")
def get_npk(
    soil_type: str,
    last_crop: str,
    residue_left: bool,
    fertilizers: dict = None,
    db: Session = Depends(get_db),
):
    """
    Estimate NPK values based on soil type, last crop, crop residue, and fertilizer history.

    :param soil_type: Type of soil (e.g., loamy, sandy, clayey).
    :param last_crop: Name of the last crop planted.
    :param residue_left: Whether crop residue was left on the field.
    :param fertilizers: Dictionary containing fertilizer contributions for N, P, and K.
    :return: Estimated NPK values in kg/hectare.
    """
    crop = db.query(Crop).filter(Crop.name == last_crop).first()
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")

    return estimate_npk(
        soil_type=soil_type,
        n=crop.n_depletion,
        p=crop.p_depletion,
        k=crop.k_depletion,
        residue_left=residue_left,
        fertilizers=fertilizers,
    )

@router.get("/initialize")
def initialize_npk(soil_type: str):
    """
    Initialize NPK values based on soil type.

    :param soil_type: Type of soil (e.g., loamy, sandy, clayey).
    :return: Initial NPK values in kg/hectare.
    """
    npk = initialize_npk_from_soil(soil_type)
    if npk == {"N": 0, "P": 0, "K": 0}:
        raise HTTPException(status_code=400, detail="Unknown soil type.")
    return {"soil_type": soil_type, "initial_npk": npk}
