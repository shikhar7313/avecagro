from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from crop_recommendation.db.database import SessionLocal
from crop_recommendation.db.models import Crop
from crop_recommendation.logic.crop_recommender import recommend_crops

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/")
def create_crop(name: str, n_depletion: float, p_depletion: float, k_depletion: float, db: Session = Depends(get_db)):
    crop = Crop(name=name, n_depletion=n_depletion, p_depletion=p_depletion, k_depletion=k_depletion)
    db.add(crop)
    db.commit()
    db.refresh(crop)
    return crop

# Predefined crop dataset
CROP_DATASET = [
    {
        "name": "wheat",
        "ideal_n": 20,
        "ideal_p": 10,
        "ideal_k": 15,
        "min_temp": 10,
        "max_temp": 25,
        "min_rainfall": 300,
        "max_rainfall": 500,
        "farming_methods": ["conventional", "organic"]
    },
    {
        "name": "rice",
        "ideal_n": 25,
        "ideal_p": 12,
        "ideal_k": 18,
        "min_temp": 20,
        "max_temp": 35,
        "min_rainfall": 800,
        "max_rainfall": 1200,
        "farming_methods": ["conventional"]
    },
    {
        "name": "maize",
        "ideal_n": 30,
        "ideal_p": 15,
        "ideal_k": 20,
        "min_temp": 15,
        "max_temp": 30,
        "min_rainfall": 400,
        "max_rainfall": 600,
        "farming_methods": ["organic"]
    }
]

@router.post("/recommend")
def recommend_crops_endpoint(
    npk: dict,
    temperature: float,
    rainfall: float,
    farming_type: str
):
    """
    Recommend crops based on NPK values, temperature, rainfall, and farming type.

    :param npk: Dictionary containing estimated N, P, and K values.
    :param temperature: Current temperature in degrees Celsius.
    :param rainfall: Current rainfall in mm.
    :param farming_type: Farming type (organic or conventional).
    :return: Top 3 recommended crops with scores and explanations.
    """
    if farming_type not in ["organic", "conventional"]:
        raise HTTPException(status_code=400, detail="Invalid farming type. Choose 'organic' or 'conventional'.")

    recommendations = recommend_crops(npk, temperature, rainfall, farming_type, CROP_DATASET)
    return {"recommendations": recommendations}
