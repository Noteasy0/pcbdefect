# utils.py - Enhanced version
import base64
import os
import uuid
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks"""
    # Remove any path components
    filename = os.path.basename(filename)
    # Remove non-alphanumeric characters except dots and underscores
    filename = re.sub(r'[^\w\.]', '_', filename)
    return filename

def validate_image_size(base64_str: str, max_size: int) -> None:
    """
    Validate image size
    
    Args:
        base64_str: Base64 encoded image
        max_size: Maximum allowed size in bytes
        
    Raises:
        ValueError: If image is too large
    """
    # Estimate size (base64 is ~33% larger than binary)
    estimated_size = len(base64_str) * 3 / 4
    
    if estimated_size > max_size:
        raise ValueError(
            f"Image too large: {estimated_size/1024/1024:.2f}MB "
            f"(max: {max_size/1024/1024:.2f}MB)"
        )

def save_base64_image(base64_str: str, prefix: str = "img") -> str:
    """
    Save base64 encoded image to disk
    
    Args:
        base64_str: Base64 encoded image (with or without data URI prefix)
        prefix: Filename prefix
        
    Returns:
        str: Path to saved image (relative)
        
    Raises:
        ValueError: If image format is invalid
    """
    try:
        # Handle data URI format
        if "," in base64_str:
            header, base64_str = base64_str.split(",", 1)
            
            # Extract image format from header
            if "image/" in header:
                format_match = re.search(r'image/(\w+)', header)
                if format_match:
                    ext = format_match.group(1)
                    if ext == "jpeg":
                        ext = "jpg"
                else:
                    ext = "jpg"
            else:
                ext = "jpg"
        else:
            ext = "jpg"
        
        # Decode base64
        try:
            imgdata = base64.b64decode(base64_str)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}")
        
        # Validate image size (basic check)
        if len(imgdata) < 100:
            raise ValueError("Image data too small, possibly corrupted")
        
        # Generate unique filename
        sanitized_prefix = sanitize_filename(prefix)
        filename = f"{sanitized_prefix}_{uuid.uuid4().hex}.{ext}"
        filepath = UPLOAD_DIR / filename
        
        # Save file
        with open(filepath, "wb") as f:
            f.write(imgdata)
        
        logger.info(f"Image saved: {filepath}")
        
        # Return relative path
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise

def now_utc() -> datetime:
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)

def parse_timewindow(
    from_ts: Optional[str],
    to_ts: Optional[str]
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse time window from ISO format strings
    
    Args:
        from_ts: Start timestamp (ISO format)
        to_ts: End timestamp (ISO format)
        
    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    if not from_ts or not to_ts:
        return None, None
    
    try:
        # Parse ISO format
        from_dt = datetime.fromisoformat(from_ts.replace('Z', '+00:00'))
        to_dt = datetime.fromisoformat(to_ts.replace('Z', '+00:00'))
        
        # Validate order
        if from_dt > to_dt:
            logger.warning("from_ts is after to_ts, swapping")
            from_dt, to_dt = to_dt, from_dt
        
        return from_dt, to_dt
        
    except Exception as e:
        logger.error(f"Error parsing time window: {e}")
        return None, None

def compute_control_chart_stats(records: List[Any]) -> Dict[str, Any]:
    """
    Compute control chart statistics from detection records
    
    Args:
        records: List of Detection ORM objects
        
    Returns:
        Dictionary containing:
        - overall_rate: Overall defect rate (0-1)
        - total: Total detections
        - defects: Total defects
        - daily: List of daily statistics
        - weekly: List of weekly statistics
    """
    if not records:
        return {
            "overall_rate": 0.0,
            "total": 0,
            "defects": 0,
            "daily": [],
            "weekly": []
        }
    
    # Daily aggregation
    daily = defaultdict(lambda: {"defect": 0, "total": 0})
    
    for r in records:
        date_key = r.timestamp.date().isoformat()
        daily[date_key]["total"] += 1
        if r.is_defect:
            daily[date_key]["defect"] += 1
    
    # Convert to list with rates
    daily_list = []
    for day in sorted(daily.keys()):
        v = daily[day]
        rate = v["defect"] / v["total"] if v["total"] > 0 else 0.0
        daily_list.append({
            "date": day,
            "defect": v["defect"],
            "total": v["total"],
            "rate": round(rate, 4)
        })
    
    # Weekly aggregation
    weekly = defaultdict(lambda: {"defect": 0, "total": 0})
    
    for r in records:
        # Get ISO week number
        year, week, _ = r.timestamp.isocalendar()
        week_key = f"{year}-W{week:02d}"
        weekly[week_key]["total"] += 1
        if r.is_defect:
            weekly[week_key]["defect"] += 1
    
    # Convert to list with rates
    weekly_list = []
    for week in sorted(weekly.keys()):
        v = weekly[week]
        rate = v["defect"] / v["total"] if v["total"] > 0 else 0.0
        weekly_list.append({
            "week": week,
            "defect": v["defect"],
            "total": v["total"],
            "rate": round(rate, 4)
        })
    
    # Overall statistics
    total = sum(v["total"] for v in daily.values())
    defects = sum(v["defect"] for v in daily.values())
    overall_rate = defects / total if total > 0 else 0.0
    
    return {
        "overall_rate": round(overall_rate, 4),
        "total": total,
        "defects": defects,
        "daily": daily_list,
        "weekly": weekly_list
    }

def calculate_control_limits(rates: List[float]) -> Dict[str, float]:
    """
    Calculate statistical control limits (UCL, LCL, CL)
    
    Args:
        rates: List of defect rates
        
    Returns:
        Dictionary with UCL, CL, LCL values
    """
    if not rates:
        return {"UCL": 0, "CL": 0, "LCL": 0}
    
    import statistics
    
    # Calculate mean (center line)
    cl = statistics.mean(rates)
    
    # Calculate standard deviation
    if len(rates) > 1:
        std = statistics.stdev(rates)
    else:
        std = 0
    
    # Calculate control limits (3 sigma)
    ucl = cl + 3 * std
    lcl = max(0, cl - 3 * std)  # Can't be negative
    
    return {
        "UCL": round(ucl, 4),
        "CL": round(cl, 4),
        "LCL": round(lcl, 4)
    }

def cleanup_old_images(days: int = 30) -> int:
    """
    Clean up old images from upload directory
    
    Args:
        days: Delete images older than this many days
        
    Returns:
        Number of files deleted
    """
    cutoff = datetime.now() - timedelta(days=days)
    deleted = 0
    
    try:
        for filepath in UPLOAD_DIR.glob("*"):
            if filepath.is_file():
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                if mtime < cutoff:
                    filepath.unlink()
                    deleted += 1
                    logger.info(f"Deleted old image: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning up images: {e}")
    
    return deleted

def get_storage_stats() -> Dict[str, Any]:
    """
    Get storage statistics for upload directory
    
    Returns:
        Dictionary with file count and total size
    """
    try:
        files = list(UPLOAD_DIR.glob("*"))
        file_count = len([f for f in files if f.is_file()])
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        return {
            "file_count": file_count,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "average_size_kb": round(total_size / file_count / 1024, 2) if file_count > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        return {"file_count": 0, "total_size_mb": 0, "average_size_kb": 0}