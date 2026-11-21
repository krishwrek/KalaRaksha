"""
KalaRaksha - Production Security & Input Validation Engine
Implements defense-in-depth protection:
1. File upload size & MIME magic-byte verification (JPEG, PNG, WEBP)
2. Image decompression bomb (decompression-bomb DoS) prevention
3. Strict filename sanitization (path traversal & shell escape prevention)
4. Sliding-window IP rate limiting
5. Security headers enforcement (CSP, HSTS, X-Content-Type-Options, etc.)
6. HTML entity escaping (XSS prevention)
7. Constant-time API key verification
"""

import os
import re
import time
import html
import hmac
import threading
from typing import Tuple, Optional, Dict, List
from PIL import Image
import io

# Limits
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15")) * 1024 * 1024
MAX_IMAGE_DIMENSION_PX = int(os.getenv("MAX_IMAGE_DIMENSION_PX", "4096"))
MAX_TOTAL_PIXELS = int(os.getenv("DECOMPRESSION_BOMB_LIMIT_PIXELS", "178956970"))  # Pillow default bomb threshold

# Ensure Pillow has strict decompression bomb protection enabled
Image.MAX_IMAGE_PIXELS = MAX_TOTAL_PIXELS

# Known magic bytes for supported formats
MAGIC_BYTES = {
    "jpeg": [
        b"\xFF\xD8\xFF\xDB",
        b"\xFF\xD8\xFF\xE0",
        b"\xFF\xD8\xFF\xE1",
        b"\xFF\xD8\xFF\xEE",
        b"\xFF\xD8\xFF\xE2",
        b"\xFF\xD8\xFF\xE3"
    ],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"]  # Checked in conjunction with 'WEBP' at offset 8
}


def sanitize_filename(filename: Optional[str]) -> str:
    """
    Sanitize filename to prevent path traversal, null-byte injection, and shell escaping.
    """
    if not filename:
        return "unnamed_wall.jpg"

    # Strip directory components (handles both POSIX and Windows separators)
    clean_name = os.path.basename(filename)
    clean_name = clean_name.replace("\\", "/").split("/")[-1]

    # Remove null bytes and control characters
    clean_name = clean_name.replace("\x00", "")
    clean_name = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", clean_name)

    # Restrict to safe alphanumeric, dots, underscores, dashes
    clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", clean_name)

    # Prevent hidden files (.htaccess, .env) or directory traversal sequences
    clean_name = clean_name.lstrip(".")
    if not clean_name:
        clean_name = "sanitized_heritage_wall.jpg"

    return clean_name[:120]  # Bounded length


def detect_file_format_from_magic_bytes(header: bytes) -> Optional[str]:
    """
    Verify actual file format using magic bytes rather than trusting file extensions or MIME headers.
    """
    if len(header) < 12:
        return None

    # Check JPEG
    if header.startswith(b"\xFF\xD8\xFF"):
        return "jpeg"

    # Check PNG
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    # Check WEBP: starts with 'RIFF' and has 'WEBP' at index 8-12
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"

    return None


def validate_image_payload(
    payload_bytes: bytes,
    original_filename: Optional[str] = None
) -> Tuple[bool, str, str]:
    """
    Perform rigorous defense-in-depth validation on uploaded image payload.

    Checks:
    1. Empty payload check
    2. Payload size ceiling (prevents memory exhaustion)
    3. Magic byte signature verification (prevents file masquerading & polyglots)
    4. Pillow structural decode check (prevents corrupt or truncated streams)
    5. Decompression bomb check (max width, height, total pixels)

    Returns:
        (is_valid: bool, sanitized_filename: str, error_message: str)
    """
    safe_filename = sanitize_filename(original_filename)

    # 1. Check empty
    if not payload_bytes or len(payload_bytes) == 0:
        return False, safe_filename, "Uploaded payload is empty."

    # 2. Check maximum size
    if len(payload_bytes) > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        return False, safe_filename, f"File size exceeds maximum allowed threshold of {max_mb:.0f} MB."

    # 3. Magic byte verification
    fmt = detect_file_format_from_magic_bytes(payload_bytes[:32])
    if not fmt:
        return False, safe_filename, "Unsupported file format. Only authentic JPEG, PNG, and WEBP image files are permitted."

    # 4. Decompression & structural verification via Pillow
    try:
        with Image.open(io.BytesIO(payload_bytes)) as img:
            img.verify()  # Verifies file integrity without reading entire raster into memory
    except Exception as e:
        return False, safe_filename, "Corrupted or invalid image stream."

    # Re-open to inspect dimensions safely (Pillow resets state after verify)
    try:
        with Image.open(io.BytesIO(payload_bytes)) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, safe_filename, "Image dimensions must be positive non-zero integers."
            if w > MAX_IMAGE_DIMENSION_PX or h > MAX_IMAGE_DIMENSION_PX:
                return False, safe_filename, f"Image dimensions ({w}x{h}) exceed maximum allowed dimension of {MAX_IMAGE_DIMENSION_PX}px."
            if (w * h) > (MAX_IMAGE_DIMENSION_PX * MAX_IMAGE_DIMENSION_PX):
                return False, safe_filename, "Image pixel volume exceeds safe processing limits (potential decompression bomb)."
    except Exception as e:
        return False, safe_filename, f"Image metadata verification failed."

    return True, safe_filename, ""


class InMemoryRateLimiter:
    """
    Thread-safe in-memory sliding-window IP rate limiter.
    """
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self.clients: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if client is within rate limit.
        Returns (is_allowed, remaining_requests).
        """
        now = time.time()
        with self.lock:
            # Clean up old timestamps
            timestamps = self.clients.get(client_ip, [])
            valid_timestamps = [t for t in timestamps if now - t < self.window_seconds]

            if len(valid_timestamps) >= self.requests_per_minute:
                self.clients[client_ip] = valid_timestamps
                return False, 0

            valid_timestamps.append(now)
            self.clients[client_ip] = valid_timestamps
            remaining = self.requests_per_minute - len(valid_timestamps)
            return True, remaining


def verify_api_key(provided_key: Optional[str]) -> bool:
    """
    Verify API key using constant-time comparison to prevent timing attacks.
    If no API_SECRET_KEY is configured in the environment, API is public.
    """
    configured_key = os.getenv("API_SECRET_KEY", "").strip()
    if not configured_key:
        return True  # Public mode
    if not provided_key:
        return False
    return hmac.compare_digest(provided_key.strip(), configured_key)


def get_production_security_headers() -> Dict[str, str]:
    """
    Return hardened HTTP security headers.
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; font-src 'self' data:;",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "X-XSS-Protection": "1; mode=block",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=()"
    }


def safe_html(text: Any) -> str:
    """
    Escape string for safe HTML rendering, neutralizing XSS vectors.
    """
    if text is None:
        return ""
    return html.escape(str(text), quote=True)
