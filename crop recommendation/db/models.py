from sqlalchemy import Column, Integer, String, Float
from crop_recommendation.db.database import Base

class Crop(Base):
    __tablename__ = "crops"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    n_depletion = Column(Float)
    p_depletion = Column(Float)
    k_depletion = Column(Float)

class Farmer(Base):
    __tablename__ = "farmers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    soil_type = Column(String)
    crop_history = Column(String)  # JSON string to store crop history
