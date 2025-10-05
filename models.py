# models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Float, Text
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import json
import pytz

tz = pytz.timezone('Asia/Bangkok')

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    detections = relationship("Detection", back_populates="user")

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String(255))
    timestamp = Column(DateTime, default=lambda: datetime.now(tz))  # <-- ใช้เวลาไทย
    is_defect = Column(Boolean, default=False)
    classes_json = Column(Text)
    bboxes_json = Column(Text)
    process_time = Column(Float)
    raw_result_json = Column(Text)

    user = relationship("User", back_populates="detections")
