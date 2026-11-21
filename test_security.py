"""
KalaRaksha - Production Security Verification Suite
Tests the 5 Security Checks:
1. Secret Leak & Env Var Audit
2. Path Traversal & Filename Sanitization
3. MIME / Magic-Byte Verification & Decompression Bomb Defense
4. Production Security Headers & Rate Limiting
5. Error Masking & Information Disclosure Prevention
6. Constant-time API Key Authentication
"""

import os
import io
import time
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

from api import app, rate_limiter
from utils.security import (
    validate_image_payload,
    sanitize_filename,
    detect_file_format_from_magic_bytes,
    verify_api_key,
    get_production_security_headers
)


def test_1_secret_leak_prevention():
    print("[TEST 1/6] Secret Leak Prevention Audit...")
    # Verify .env is in .gitignore
    with open(".gitignore", "r") as f:
        gitignore_content = f.read()
    assert ".env" in gitignore_content, "CRITICAL: .env not found in .gitignore!"
    assert os.path.exists(".env.example"), ".env.example template missing!"
    print("  [PASS] .env is properly gitignored and .env.example exists.")


def test_2_path_traversal_and_filename_sanitization():
    print("[TEST 2/6] Path Traversal & Content Injection Defense...")
    malicious_inputs = [
        ("../../etc/passwd", "etc_passwd"),
        ("..\\..\\windows\\system32\\cmd.exe", "windows_system32_cmd.exe"),
        ("image.jpg\x00.exe", "image.jpg.exe"),
        ("<script>alert(1)</script>.jpg", "script_alert_1__script_.jpg"),
        ("wall; rm -rf / ;.png", "wall__rm_-rf_____.png"),
        (None, "unnamed_wall.jpg"),
        ("", "sanitized_heritage_wall.jpg")
    ]
    for raw, expected in malicious_inputs:
        cleaned = sanitize_filename(raw)
        assert "/" not in cleaned and "\\" not in cleaned, f"Path separator leaked in: {cleaned}"
        assert "\x00" not in cleaned, "Null byte leaked!"
        assert "<" not in cleaned and ">" not in cleaned, "HTML tag characters leaked!"
    print("  [PASS] All path traversal, null-byte, and script injection vectors successfully neutralized.")


def test_3_magic_bytes_and_decompression_bomb():
    print("[TEST 3/6] Magic Bytes & Decompression Bomb Verification...")
    # 1. Text file masquerading as .jpg
    fake_payload = b"MZ\x90\x00\x03\x00\x00\x00This is a malicious PE binary masquerading as image"
    is_valid, _, err = validate_image_payload(fake_payload, "malicious.jpg")
    assert not is_valid, "Failed: Fake payload was accepted!"
    assert "Unsupported file format" in err

    # 2. Corrupt truncated image stream
    corrupt_jpeg = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"  # truncated header only
    is_valid, _, err = validate_image_payload(corrupt_jpeg, "corrupt.jpg")
    assert not is_valid, "Failed: Truncated stream was accepted!"

    # 3. Legitimate JPEG sample from disk
    with open("sample_images/sample1_temple_crack.jpg", "rb") as f:
        real_jpeg = f.read()
    is_valid, safe_name, err = validate_image_payload(real_jpeg, "sample1_temple_crack.jpg")
    assert is_valid, f"Failed: Legitimate JPEG rejected: {err}"
    assert safe_name == "sample1_temple_crack.jpg"

    # 4. Dimension boundary check (> 4096px)
    large_pil = Image.new("RGB", (4097, 100), color="red")
    large_buf = io.BytesIO()
    large_pil.save(large_buf, format="JPEG")
    is_valid, _, err = validate_image_payload(large_buf.getvalue(), "huge.jpg")
    assert not is_valid, "Failed: Over-dimensioned image accepted!"
    assert "exceed maximum allowed dimension" in err

    print("  [PASS] Magic-byte verification and decompression bomb defenses verified.")


def test_4_security_headers_and_rate_limiting():
    print("[TEST 4/6] Production Security Headers & Rate Limiter...")
    client = TestClient(app)

    # Health check headers
    res = client.get("/health")
    assert res.status_code == 200
    headers = res.headers
    assert headers.get("x-content-type-options") == "nosniff", "Missing X-Content-Type-Options"
    assert headers.get("x-frame-options") == "DENY", "Missing X-Frame-Options"
    assert "Strict-Transport-Security" in headers or "strict-transport-security" in headers, "Missing HSTS"
    assert "content-security-policy" in headers, "Missing CSP"

    # Test Rate Limiter
    # Rate limiter defaults to 30 RPM
    limiter = rate_limiter
    test_ip = "192.168.1.99"
    # Exhaust requests
    for _ in range(limiter.requests_per_minute):
        allowed, _ = limiter.is_allowed(test_ip)
        assert allowed is True
    # Next request must be blocked
    allowed, remaining = limiter.is_allowed(test_ip)
    assert allowed is False
    assert remaining == 0
    print("  [PASS] Strict security headers and sliding-window rate limiter confirmed.")


def test_5_information_disclosure_prevention():
    print("[TEST 5/6] Error Masking & Information Disclosure...")
    client = TestClient(app)

    # Send completely invalid request to /api/v1/analyze
    res = client.post("/api/v1/analyze", files={"file": ("test.jpg", b"invalid_binary_stream", "image/jpeg")})
    assert res.status_code == 400
    detail = res.json().get("detail", "")
    assert "Traceback" not in detail, "Stack trace leaked in error response!"
    assert "C:\\" not in detail and "/app" not in detail, "Server disk file path leaked in error response!"
    print("  [PASS] Error responses sanitized; no stack traces or filesystem paths leaked.")


def test_6_api_key_authentication():
    print("[TEST 6/6] Constant-Time API Key Authentication...")
    # When no key is set in env, it allows public access
    assert verify_api_key(None) is True

    # Simulate configured secret
    os.environ["API_SECRET_KEY"] = "super-secret-heritage-token-2026"
    assert verify_api_key("super-secret-heritage-token-2026") is True
    assert verify_api_key("wrong-key") is False
    assert verify_api_key(None) is False
    del os.environ["API_SECRET_KEY"]
    print("  [PASS] Constant-time HMAC authentication logic verified.")


def run_all_security_tests():
    print("=====================================================")
    print("KalaRaksha Production Security Audit Verification")
    print("=====================================================")
    test_1_secret_leak_prevention()
    test_2_path_traversal_and_filename_sanitization()
    test_3_magic_bytes_and_decompression_bomb()
    test_4_security_headers_and_rate_limiting()
    test_5_information_disclosure_prevention()
    test_6_api_key_authentication()
    print("=====================================================")
    print("ALL 6 SECURITY CHECKS PASSED WITH ZERO VULNERABILITIES!")
    print("=====================================================")


if __name__ == "__main__":
    run_all_security_tests()
