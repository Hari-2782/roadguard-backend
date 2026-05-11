"""
Cloudinary helper for RoadGuard evidence storage.
Uploads images/videos to Cloudinary and returns public URLs.
Falls back to local storage if Cloudinary is not configured.
"""
import os
import logging
import tempfile

logger = logging.getLogger("CloudinaryHelper")

# Cloudinary config from environment variables
CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "")
API_KEY = os.environ.get("CLOUDINARY_API_KEY", "")
API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "")

_cloudinary_configured = False
_cloudinary_initialized = False

def _init_cloudinary():
    """Initialize Cloudinary SDK if credentials are available."""
    global _cloudinary_configured, _cloudinary_initialized
    if _cloudinary_initialized:
        return _cloudinary_configured
    _cloudinary_initialized = True

    if not all([CLOUD_NAME, API_KEY, API_SECRET]):
        logger.warning("Cloudinary not configured (missing env vars). Using local storage.")
        return False

    try:
        import cloudinary
        cloudinary.config(
            cloud_name=CLOUD_NAME,
            api_key=API_KEY,
            api_secret=API_SECRET,
            secure=True
        )
        _cloudinary_configured = True
        logger.info(f"Cloudinary initialized (cloud: {CLOUD_NAME})")
        return True
    except ImportError:
        logger.warning("cloudinary package not installed. Using local storage.")
        return False
    except Exception as e:
        logger.error(f"Cloudinary init error: {e}")
        return False


# In-memory cache: filename -> cloudinary URL
_url_cache = {}


def upload_evidence_image(frame, filename, folder="roadguard/evidence"):
    """
    Upload an OpenCV frame (numpy array) to Cloudinary as a JPEG.
    
    Args:
        frame: OpenCV BGR image (numpy array)
        filename: desired filename (e.g., "HZ_20260511_120000.jpg")
        folder: Cloudinary folder path
    
    Returns:
        str: The public_id or filename to store in DB.
             If Cloudinary is active, also caches the URL.
    """
    if not _init_cloudinary():
        return None  # Caller should fall back to local cv2.imwrite

    try:
        import cloudinary.uploader
        import cv2
        
        # Encode frame to JPEG bytes
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            logger.error("Failed to encode frame to JPEG")
            return None

        # Upload bytes to Cloudinary
        public_id = f"{folder}/{os.path.splitext(filename)[0]}"
        result = cloudinary.uploader.upload(
            buffer.tobytes(),
            public_id=public_id,
            resource_type="image",
            overwrite=True,
            format="jpg"
        )
        
        url = result.get("secure_url", "")
        if url:
            _url_cache[filename] = url
            logger.info(f"Uploaded evidence: {filename} -> {url[:80]}...")
            return filename  # Return the filename key (DB stores this)
        else:
            logger.error(f"Cloudinary upload returned no URL for {filename}")
            return None

    except Exception as e:
        logger.error(f"Cloudinary upload error for {filename}: {e}")
        return None


def upload_video(file_path, filename, folder="roadguard/videos"):
    """
    Upload a video file to Cloudinary.
    
    Args:
        file_path: local path to the video file
        filename: desired filename
        folder: Cloudinary folder path
    
    Returns:
        str: Cloudinary URL or None
    """
    if not _init_cloudinary():
        return None

    try:
        import cloudinary.uploader
        
        public_id = f"{folder}/{os.path.splitext(filename)[0]}"
        result = cloudinary.uploader.upload_large(
            file_path,
            public_id=public_id,
            resource_type="video",
            chunk_size=6000000  # 6MB chunks
        )
        
        url = result.get("secure_url", "")
        if url:
            _url_cache[filename] = url
            logger.info(f"Uploaded video: {filename} -> {url[:80]}...")
            return url
        return None

    except Exception as e:
        logger.error(f"Cloudinary video upload error for {filename}: {e}")
        return None


def get_evidence_url(filename):
    """
    Get the public URL for an evidence file.
    
    Returns:
        str: Full Cloudinary URL if available, else None (caller serves from local disk)
    """
    # Check cache first
    if filename in _url_cache:
        return _url_cache[filename]
    
    if not _init_cloudinary():
        return None

    try:
        import cloudinary.utils
        # Build URL from cloud name + public_id
        public_id = f"roadguard/evidence/{os.path.splitext(filename)[0]}"
        url = cloudinary.utils.cloudinary_url(public_id, format="jpg", secure=True)[0]
        _url_cache[filename] = url
        return url
    except Exception:
        return None


def is_cloudinary_active():
    """Check if Cloudinary is configured and ready."""
    return _init_cloudinary()
