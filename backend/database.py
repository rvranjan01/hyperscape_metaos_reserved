import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker  # <-- Updated import here

# Database connection URL pointing to your Docker PostgreSQL container
from config import DATABASE_URL  # Import the DATABASE_URL from config.py

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Scan(Base):
  __tablename__ = "scans"

  id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
  device_name = Column(String)
  created_at = Column(DateTime, default=datetime.utcnow)
  status = Column(String, default="uploading")
  total_frames = Column(Integer)
  uploaded_frames = Column(Integer, default=0)

class Job(Base):
  __tablename__ = "jobs"

  id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
  scan_id = Column(String, ForeignKey("scans.id"))
  stage = Column(String, default="colmap")  # colmap, splat_training, complete
  status = Column(String, default="queued")  # queued, running, done, failed
  created_at = Column(DateTime, default=datetime.utcnow)
  
class Asset(Base):
  __tablename__ = "assets"

  id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
  scan_id = Column(String, ForeignKey("scans.id"))
  file_type = Column(String)  # splat, thumbnail, point_cloud
  s3_key = Column(String)
  created_at = Column(DateTime, default=datetime.utcnow)
# Generate the 'scans' table inside PostgreSQL automatically
Base.metadata.create_all(bind=engine)