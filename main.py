# main.py - Enhanced version
import time, json, os, io
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
import secrets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import base64
import pdfkit
from io import BytesIO
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database import SessionLocal, engine, Base
from models import User, Detection
from collections import Counter
from utils import (
    save_base64_image, 
    compute_control_chart_stats, 
    parse_timewindow, 
    now_utc,
    validate_image_size,
    sanitize_filename
)

# YOLOv8 (ultralytics)
from ultralytics import YOLO
import logging
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
MODEL_PATH = os.getenv("YOLO_MODEL", "best.pt")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024))  # 10MB
SESSION_EXPIRE_HOURS = int(os.getenv("SESSION_EXPIRE_HOURS", 24))
REPORT_DIR = os.path.join("reports")
REPORT_IMAGES_DIR = os.path.join(REPORT_DIR, "images")


# Global model instance
model: Optional[YOLO] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global model
    # Startup
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created")
        
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info(f"YOLO model loaded: {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")

app = FastAPI(
    title="Defect Detection System",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# Password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Security
security = HTTPBearer(auto_error=False)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication dependency
def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Get current user from session"""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    
    try:
        user = db.query(User).filter(User.id == int(user_id)).first()
        return user
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return None

def require_auth(request: Request, db: Session = Depends(get_db)) -> User:
    """Require authentication"""
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return user

# ============================================================================
# Authentication Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to login"""
    return RedirectResponse(url="/login")

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page"""
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(..., min_length=3, max_length=50),
    password: str = Form(..., min_length=6),
    db: Session = Depends(get_db)
):
    """Register new user"""
    # Validate username
    username = username.strip().lower()
    if not username.isalnum():
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Username must be alphanumeric"}
        )
    
    # Check if username exists
    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Username already exists"}
        )
    
    try:
        # Hash password and create user
        hashed = pwd_context.hash(password)
        user = User(username=username, password=hashed)
        db.add(user)
        db.commit()
        db.refresh(user)
        
        logger.info(f"New user registered: {username}")
        return RedirectResponse(url="/login?registered=1", status_code=303)
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.rollback()
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Registration failed"}
        )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, registered: int = 0):
    """Login page"""
    message = "Registration successful! Please login." if registered else None
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "message": message}
    )

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Login user"""
    username = username.strip().lower()
    
    # Find user
    user = db.query(User).filter(User.username == username).first()
    
    # Verify password
    if not user or not pwd_context.verify(password, user.password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password"}
        )
    
    # Set session
    request.session["user_id"] = user.id
    logger.info(f"User logged in: {username}")
    
    return RedirectResponse(url="/dashboard", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    """Logout user"""
    request.session.clear()
    return RedirectResponse(url="/login")

# ============================================================================
# Dashboard Routes
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Dashboard page"""
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login")
    
    # Recent detections
    records = (
        db.query(Detection)
        .filter(Detection.user_id == user.id)
        .order_by(Detection.timestamp.desc())
        .limit(20)
        .all()
    )
    
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "user": user, "recent": records}
    )

@app.get("/api/dashboard/data/{user_id}")
async def api_dashboard_data(user_id: int, db: Session = Depends(get_db)):
    """Get dashboard data"""
    records = (
        db.query(Detection)
        .filter(Detection.user_id == user_id)
        .order_by(Detection.timestamp.desc())
        .limit(100)
        .all()
    )
    
    stats = compute_control_chart_stats(records)
    
    # Calculate top classes
    from collections import Counter
    class_counter = Counter()
    for r in records:
        try:
            classes = json.loads(r.classes_json or "{}")
            class_counter.update(classes)
        except:
            pass
    
    top_classes = class_counter.most_common(10)
    
    # Recent detections
    recent = [
        {
            "timestamp": r.timestamp.isoformat(),
            "is_defect": r.is_defect,
            "image_path": r.image_path,
            "process_time": round(r.process_time, 3) if r.process_time else 0,
            "detections": json.loads(r.raw_result_json).get("boxes", [])
        }
        for r in records[:20]
    ]
    
    return {
        "stats": stats,
        "top_classes": top_classes,
        "recent": recent
    }

@app.get("/api/dashboard/control_chart/{user_id}")
async def api_dashboard_control_chart(user_id: int, db: Session = Depends(get_db)):
    """Generate control chart image"""
    records = db.query(Detection).filter(Detection.user_id == user_id).all()
    stats = compute_control_chart_stats(records)
    chart_data = stats.get("daily", [])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    if not chart_data:
        ax.text(0.5, 0.5, "No data available", 
                ha="center", va="center", fontsize=14)
    else:
        # แปลง date string เป็น datetime objects
        dates = [datetime.datetime.fromisoformat(d["date"]) for d in chart_data]
        rates = [d.get("rate", 0) * 100 for d in chart_data]

        # Plot line
        ax.plot(dates, rates, marker='o', color='#2563eb', linewidth=2, 
                markersize=6, label='Defect Rate')

        # Threshold line
        ax.axhline(y=10, color='#dc2626', linestyle='--', 
                   linewidth=2, label="Threshold (10%)")

        # Styling
        ax.set_xlabel("Date", fontsize=11, fontweight='bold')
        ax.set_ylabel("Defect Rate (%)", fontsize=11, fontweight='bold')
        ax.set_title("Daily Defect Rate Control Chart", 
                     fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', frameon=True, shadow=True)

        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Tight layout
        fig.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

# ============================================================================
# Detection API
# ============================================================================

@app.post("/api/dashboard/detect_live")
async def dashboard_detect_live(payload: Dict[str, Any], db: Session = Depends(get_db)):
    """Live detection from dashboard"""
    return await api_detect(payload, db)

@app.post("/api/detect")
async def api_detect(payload: Dict[str, Any], db: Session = Depends(get_db)):
    """
    Detect defects from base64 image
    
    Payload:
    {
        "user_id": 1,
        "image_base64": "data:image/png;base64,...",
        "conf_thres": 0.25
    }
    
    Returns:
    {
        "detections": [...],
        "is_defect": bool,
        "process_time": float,
        "class_counts": {...},
        "detection_id": int
    }
    """
    # Validate payload
    if "image_base64" not in payload:
        raise HTTPException(
            status_code=400,
            detail="image_base64 is required"
        )
    
    user_id = payload.get("user_id")
    b64_image = payload["image_base64"]
    conf_thres = float(payload.get("conf_thres", 0.25))
    
    # Validate confidence threshold
    if not 0 <= conf_thres <= 1:
        raise HTTPException(
            status_code=400,
            detail="conf_thres must be between 0 and 1"
        )
    
    # Validate image size
    try:
        validate_image_size(b64_image, MAX_IMAGE_SIZE)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Check model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Detection model not loaded"
        )
    
    try:
        # Save image
        img_path = save_base64_image(b64_image, prefix=f"user{user_id or 'anon'}")
        
        # Run inference
        start_time = time.time()
        results = model.predict(source=img_path, conf=conf_thres, imgsz=640, verbose=False)
        elapsed = time.time() - start_time
        
        # Parse results
        r = results[0]
        detections = []
        class_counts = {}
        bboxes = []
        is_defect = False
        
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            is_defect = True
            
            for box in r.boxes:
                # Extract data
                xyxy = box.xyxy.tolist()[0] if hasattr(box.xyxy, "tolist") else list(box.xyxy)
                cls_idx = int(box.cls.tolist()[0]) if hasattr(box.cls, "tolist") else int(box.cls)
                conf = float(box.conf.tolist()[0]) if hasattr(box.conf, "tolist") else float(box.conf)
                
                # Get class name
                class_name = model.names.get(cls_idx, f"class_{cls_idx}")
                
                # Add to results
                detections.append({
                    "class": class_name,
                    "conf": round(conf, 4),
                    "bbox": [round(float(x), 2) for x in xyxy]
                })
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                bboxes.append([float(x) for x in xyxy])
        
        # Save to database
        detection = Detection(
            user_id=user_id,
            image_path=img_path,
            is_defect=is_defect,
            classes_json=json.dumps(class_counts),
            bboxes_json=json.dumps(bboxes),
            process_time=elapsed,
            raw_result_json=json.dumps({"boxes": detections})
        )
        db.add(detection)
        db.commit()
        db.refresh(detection)
        
        logger.info(f"Detection completed: user={user_id}, defect={is_defect}, time={elapsed:.3f}s")
        
        return {
            "detections": detections,
            "is_defect": is_defect,
            "class_counts": class_counts,
            "process_time": round(elapsed, 3),
            "detection_id": detection.id
        }
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

# ============================================================================
# Statistics API
# ============================================================================

@app.get("/api/stats/user/{user_id}")
async def api_user_stats(
    user_id: int,
    from_ts: Optional[str] = None,
    to_ts: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get user statistics for time range"""
    from_dt, to_dt = parse_timewindow(from_ts, to_ts)
    
    # Build query
    query = db.query(Detection).filter(Detection.user_id == user_id)
    
    if from_dt and to_dt:
        query = query.filter(
            Detection.timestamp >= from_dt,
            Detection.timestamp <= to_dt
        )
    
    records = query.all()
    stats = compute_control_chart_stats(records)
    
    # Top classes
    from collections import Counter
    class_counter = Counter()
    for r in records:
        try:
            classes = json.loads(r.classes_json or "{}")
            class_counter.update(classes)
        except:
            pass
    
    stats["top_classes"] = class_counter.most_common(10)
    stats["time_range"] = {
        "from": from_ts or "all",
        "to": to_ts or "all"
    }
    
    return stats

@app.get("/api/stats/daily")
async def api_stats_daily(db: Session = Depends(get_db)):
    """Get daily statistics for all users"""
    records = db.query(Detection).all()
    stats = compute_control_chart_stats(records)
    return stats

@app.post("/api/control/check")
async def api_control_check(payload: Dict[str, Any], db: Session = Depends(get_db)):
    """
    Check control rules
    
    Payload:
    {
        "user_id": int,
        "from": str (ISO datetime),
        "to": str (ISO datetime),
        "threshold_percent": float (default: 10)
    }
    """
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    from_ts = payload.get("from")
    to_ts = payload.get("to")
    threshold = float(payload.get("threshold_percent", 10))
    
    # Get records
    from_dt, to_dt = parse_timewindow(from_ts, to_ts)
    query = db.query(Detection).filter(Detection.user_id == user_id)
    
    if from_dt and to_dt:
        query = query.filter(
            Detection.timestamp >= from_dt,
            Detection.timestamp <= to_dt
        )
    
    records = query.all()
    
    if not records:
        return {
            "defect_percent": 0,
            "threshold": threshold,
            "recommendation": "NO_DATA",
            "message": "No detection records found for this period"
        }
    
    stats = compute_control_chart_stats(records)
    defect_pct = stats["overall_rate"] * 100
    
    # Determine recommendation
    if defect_pct >= threshold:
        recommendation = "REPAIR_RECOMMENDED"
        message = f"Defect rate ({defect_pct:.2f}%) exceeds threshold ({threshold}%)"
    else:
        recommendation = "OK"
        message = f"Defect rate ({defect_pct:.2f}%) is within acceptable range"
    
    return {
        "defect_percent": round(defect_pct, 2),
        "threshold": threshold,
        "recommendation": recommendation,
        "message": message,
        "stats": stats
    }

# Color map for classes (server-side)
CLASS_COLOR = {
    "spur": (37, 99, 235),           # blue (#2563eb)
    "open": (239, 68, 68),           # red (#ef4444)
    "hole_breakout": (17, 24, 39),   # black-ish (#111827)
    "foreign_object": (16, 185, 129) # green (#10b981)
}

def rgb_tuple_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def annotate_image_save(img_path: str, detections: List[Dict[str, Any]], out_path: str):
    """Draw bounding boxes (no labels) onto image and save to out_path (uses PIL)"""
    if not PIL_AVAILABLE:
        # If PIL not available, just copy original
        try:
            with open(img_path, "rb") as fr, open(out_path, "wb") as fw:
                fw.write(fr.read())
            return
        except Exception as e:
            logger.error(f"Failed to copy image for report: {e}")
            return

    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        iw, ih = img.size

        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            # Coordinates may come from model scaled to original (we saved the model image path), assume they're in pixel units
            color = CLASS_COLOR.get(det.get("class"), (255, 0, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        img.save(out_path, format="JPEG", quality=90)
    except Exception as e:
        logger.error(f"Annotate image error: {e}")
        # fallback: copy original
        try:
            with open(img_path, "rb") as fr, open(out_path, "wb") as fw:
                fw.write(fr.read())
        except Exception as ex:
            logger.error(f"Fallback copy error: {ex}")

# New endpoint: folder detect (upload up to 20 images from client)
@app.post("/api/detect_folder")
async def api_detect_folder(user_id: int = Form(...), files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """
    Accept multiple files (from client folder selection), up to 20.
    Returns JSON summary and URL to HTML report saved in /reports.
    """
    MAX_FILES = 20
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Max {MAX_FILES} files allowed")

    if model is None:
        raise HTTPException(status_code=503, detail="Detection model not loaded")

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    report_name = f"report_{user_id}_{timestamp}.html"
    report_folder_images = os.path.join(REPORT_IMAGES_DIR, f"{user_id}_{timestamp}")
    os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    summary = {
        "user_id": user_id,
        "total_images": 0,
        "images": [],  # list of dicts: {filename, saved_path, is_defect, detections: [...]}
        "total_defect_positions": 0
    }

    for uf in files:
        try:
            filename = sanitize_filename(uf.filename)
            # Save uploaded to temp images dir
            saved_path = os.path.join("reports", "images", f"{user_id}_{timestamp}")
            os.makedirs(saved_path, exist_ok=True)
            disk_path = os.path.join(saved_path, f"{timestamp}_{filename}")
            with open(disk_path, "wb") as f:
                content = await uf.read()
                f.write(content)

            # Run inference
            start = time.time()
            results = model.predict(source=disk_path, conf=0.25, imgsz=640, verbose=False)
            elapsed = time.time() - start
            r = results[0]

            detections = []
            bboxes = []
            is_defect = False

            if hasattr(r, "boxes") and len(r.boxes) > 0:
                is_defect = True
                for box in r.boxes:
                    xyxy = box.xyxy.tolist()[0] if hasattr(box.xyxy, "tolist") else list(box.xyxy)
                    cls_idx = int(box.cls.tolist()[0]) if hasattr(box.cls, "tolist") else int(box.cls)
                    conf = float(box.conf.tolist()[0]) if hasattr(box.conf, "tolist") else float(box.conf)
                    class_name = model.names.get(cls_idx, f"class_{cls_idx}")

                    det_item = {
                        "class": class_name,
                        "conf": round(conf, 4),
                        "bbox": [round(float(x), 2) for x in xyxy]
                    }
                    detections.append(det_item)
                    bboxes.append([float(x) for x in xyxy])

            # Save detection record in DB (optional)
            try:
                detection = Detection(
                    user_id=user_id,
                    image_path=disk_path,
                    is_defect=is_defect,
                    classes_json=json.dumps({d["class"]: 1 for d in detections}) if detections else "{}",
                    bboxes_json=json.dumps(bboxes),
                    process_time=elapsed,
                    raw_result_json=json.dumps({"boxes": detections})
                )
                db.add(detection)
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning(f"DB save failed for {disk_path}: {e}")

            # Annotate & save image for report
            annotated_name = f"annot_{timestamp}_{filename}"
            annotated_path = os.path.join(report_folder_images, annotated_name)
            annotate_image_save(disk_path, detections, annotated_path)

            # add to summary
            summary["images"].append({
                "filename": filename,
                "saved_path": disk_path,
                "annotated_rel": f"images/{user_id}_{timestamp}/{annotated_name}",
                "is_defect": is_defect,
                "detections": detections
            })
            summary["total_images"] += 1
            summary["total_defect_positions"] += sum([1 for d in detections])
        except Exception as e:
            logger.error(f"Error processing file {uf.filename}: {e}")
            # continue to next file
            continue
    from collections import Counter

    defect_counter = Counter()
    for img in summary['images']:
        for d in img.get('detections', []):
            defect_counter[d['class']] += 1

    defect_classes = list(defect_counter.keys())
    defect_counts = [defect_counter[c] for c in defect_classes]        
    # Build a simple HTML report and save to reports
    report_html_parts = []
    report_html_parts.append(f"""
    <html>
    <head>
    <meta charset='utf-8'>
    <title>PCB Defect Analysis Report - {timestamp}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
    body {{
    font-family: 'Calibri', 'Times New Roman', serif;
    background-color: #ffffff;
    color: #000000;
    margin: 50px;
    line-height: 1.5;
    }}

    h1 {{
    text-align: center;
    font-size: 22pt;
    border-bottom: 2px solid #000;
    padding-bottom: 10px;
    margin-bottom: 30px;
    }}

    h2 {{
    font-size: 14pt;
    margin-top: 40px;
    margin-bottom: 10px;
    text-decoration: underline;
    }}

    p, .report-meta {{
    font-size: 12pt;
    text-align: justify;
    }}

    .table-container {{
    text-align: center;
    margin: 30px auto;
    }}

    table {{
    width: 90%;
    border-collapse: collapse;
    margin: 0 auto;
    font-size: 11pt;
    background-color: #fff;
    }}

    th, td {{
    border: 1px solid #bfbfbf;
    padding: 8px 12px;
    text-align: center;
    vertical-align: middle;
    }}

    th {{
    background-color: #f2f2f2;
    font-weight: bold;
    }}

    tr.defect-row {{
    background-color: #f9f9f9;
    }}

    img.report-img {{
    max-width: 160px;
    border: 1px solid #ccc;
    border-radius: 4px;
    display: block;
    margin: 5px auto;
    }}

    .legend {{
    margin: 20px 0;
    border: 1px solid #bfbfbf;
    padding: 10px 15px;
    background-color: #f9f9f9;
    width: 90%;
    margin-left: auto;
    margin-right: auto;
    }}

    .legend-title {{
    font-weight: bold;
    margin-bottom: 5px;
    }}

    .legend-item {{
    font-size: 11pt;
    margin-bottom: 3px;
    }}

    .legend-color {{
    display: inline-block;
    width: 14px;
    height: 14px;
    margin-right: 8px;
    border: 1px solid #888;
    vertical-align: middle;
    }}

    .footer {{
    text-align: center;
    font-size: 10pt;
    color: #555;
    margin-top: 40px;
    border-top: 1px solid #ccc;
    padding-top: 10px;
    }}

    .chart-section {{
    text-align: center;
    margin: 60px auto;
    width: 70%;
    }}

    canvas {{
    max-width: 520px;
    margin-top: 20px;
    }}
    </style>
    </head>

    <body>
    <h1>PCB Defect Analysis Report</h1>

    <p class="report-meta" style="text-align:center;">
    Generated: {datetime.datetime.utcnow().isoformat()} UTC | Local Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
    User ID: {user_id}
    </p>

    <div class="legend">
    <div class="legend-title">Color Legend (Defect Classification)</div>
    <div class="legend-item"><span class="legend-color" style="background-color:#3b82f6;"></span> Spur (สีน้ำเงิน)</div>
    <div class="legend-item"><span class="legend-color" style="background-color:#ef4444;"></span> Open (สีแดง)</div>
    <div class="legend-item"><span class="legend-color" style="background-color:#111827;"></span> Hole Breakout (สีดำ)</div>
    <div class="legend-item"><span class="legend-color" style="background-color:#22c55e;"></span> Foreign Object (สีเขียว)</div>
    </div>

    <h2>1. Inspection Summary</h2>
    <p style="text-align:center;">
    <b>Total Images:</b> {summary['total_images']} &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>Total Defect Positions:</b> {summary['total_defect_positions']}
    </p>

    <div class="table-container">
    <table>
    <tr>
        <th>No.</th>
        <th>Image & Detected Area</th>
        <th>Filename</th>
        <th>Status</th>
    </tr>
    """)

    # ตารางข้อมูล
    for i, img in enumerate(summary['images']):
        row_class = 'defect-row' if img['is_defect'] else ''
        detections_html = "<ul style='text-align:left; margin:0; padding-left:20px; font-size:10pt;'>" + "".join(
            [f"<li>{d['class']} — bbox: {d['bbox']}</li>" for d in img['detections']]
        ) + "</ul>" if img['detections'] else "<p style='font-size:10pt;'>No Defect</p>"
        
        report_html_parts.append(f"""
        <tr class="{row_class}">
        <td>{i+1}</td>
        <td>
            <img src="./{img['annotated_rel']}" class="report-img"/>
            {detections_html}
        </td>
        <td>{img['filename']}</td>
        <td>{'DEFECT' if img['is_defect'] else 'OK'}</td>
        </tr>
        """)

    # นับจำนวน defect แยกตามประเภท
    spur_count = sum(d['class'] == 'spur' for img in summary['images'] for d in img['detections'])
    open_count = sum(d['class'] == 'open' for img in summary['images'] for d in img['detections'])
    hole_count = sum(d['class'] == 'hole_breakout' for img in summary['images'] for d in img['detections'])
    foreign_count = sum(d['class'] == 'foreign_object' for img in summary['images'] for d in img['detections'])

    report_html_parts.append(f"""
    </table>
    </div>

    <h2>2. Statistical Visualization</h2>
    <p style="text-align:center;">
    The chart below shows the number of detected defects for each defect type across all inspected images.
    </p>

    <div class="chart-section">
    <canvas id="barChart"></canvas>
    </div>

    <div class="footer">
    © {datetime.datetime.now().year} PCB Quality Analysis System – Automated Inspection Report
    </div>

    <script>
    new Chart(document.getElementById('barChart'), {{
    type: 'bar',
    data: {{
        labels: ['Spur', 'Open', 'Hole Breakout', 'Foreign Object'],
        datasets: [{{
        label: 'Number of Defects',
        data: [{spur_count}, {open_count}, {hole_count}, {foreign_count}],
        backgroundColor: [
            'rgba(59,130,246,0.8)',
            'rgba(239,68,68,0.8)',
            'rgba(17,24,39,0.8)',
            'rgba(34,197,94,0.8)'
        ],
        borderColor: [
            'rgba(59,130,246,1)',
            'rgba(239,68,68,1)',
            'rgba(17,24,39,1)',
            'rgba(34,197,94,1)'
        ],
        borderWidth: 1
        }}]
    }},
    options: {{
        scales: {{
        y: {{
            beginAtZero: true,
            title: {{
            display: true,
            text: 'จำนวนข้อบกพร่อง (ครั้ง)'
            }}
        }},
        x: {{
            title: {{
            display: true,
            text: 'ประเภทข้อบกพร่อง'
            }}
        }}
        }},
        plugins: {{
        legend: {{
            display: false
        }},
        title: {{
            display: true,
            text: 'Defect Distribution by Type',
            font: {{ size: 16 }}
        }}
        }}
    }}
    }});
    </script>

    </body>
    </html>
    """)

    report_html = "\n".join(report_html_parts)
    report_path = os.path.join(REPORT_DIR, report_name)
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_html)

    # คืนค่า summary
    report_url = f"/reports/{report_name}"
    return {
        "summary": {
            "total_images": summary["total_images"],
            "total_defect_positions": summary["total_defect_positions"],
            "images": [
                {"filename": i["filename"], "is_defect": i["is_defect"], "annotated": i["annotated_rel"]}
                for i in summary["images"]
            ]
        },
        "report_url": report_url
    }


# Optional: endpoint to fetch report file (served by StaticFiles mounted at /reports)
@app.get("/api/report/{report_name}", response_class=HTMLResponse)
async def get_report(report_name: str):
    # Security: sanitize name
    safe = sanitize_filename(report_name)
    path = os.path.join(reports, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.datetime.now().isoformat()
    }