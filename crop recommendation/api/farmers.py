from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from crop_recommendation.db.database import SessionLocal
from crop_recommendation.db.models import Farmer
from crop_recommendation.logic.crop_rotation import suggest_crop_rotation

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/")
def create_farmer(name: str, soil_type: str, crop_history: str, db: Session = Depends(get_db)):
    farmer = Farmer(name=name, soil_type=soil_type, crop_history=crop_history)
    db.add(farmer)
    db.commit()
    db.refresh(farmer)
    return farmer

@router.get("/{farmer_id}/crop-rotation")
def get_crop_rotation(farmer_id: int, db: Session = Depends(get_db)):
    """
    Suggest crop rotations based on the farmer's crop history.

    :param farmer_id: ID of the farmer.
    :param db: Database session.
    :return: Suggested crop rotations.
    """
    farmer = db.query(Farmer).filter(Farmer.id == farmer_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found.")

    crop_history = farmer.crop_history.split(",") if farmer.crop_history else []
    return suggest_crop_rotation(crop_history)
