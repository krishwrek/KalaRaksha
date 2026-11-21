"""
KalaRaksha - Server Log & Correlation ID Inspector
Helper utility for developers and administrators to quickly diagnose client errors,
inspect correlation IDs, and review system security events.

Usage:
    python inspect_logs.py                     # Shows latest system status & security health
    python inspect_logs.py --id <UUID>         # Searches for specific error correlation ID
    python inspect_logs.py --clean-outputs     # Cleans temporary exports older than 7 days
"""

import os
import sys
import argparse
import datetime
import glob


def inspect_correlation_id(correlation_id: str):
    print(f"Searching for Correlation ID: {correlation_id}...")
    found = False
    log_files = glob.glob("*.log") + glob.glob("logs/*.log")

    for log_path in log_files:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if correlation_id in line:
                        print(f"[{log_path}] {line.strip()}")
                        found = True

    if not found:
        print(f"No logged server errors matching Correlation ID '{correlation_id}' were found.")
        print("Note: In development mode, FastAPI errors are displayed directly in the terminal console running uvicorn.")


def show_system_health():
    print("=====================================================")
    print("KalaRaksha System Status & Security Audit Summary")
    print("=====================================================")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.path.abspath('.')}")

    # Check .env
    has_env = os.path.exists(".env")
    print(f"Configuration File (.env): {'[PRESENT & LOADED]' if has_env else '[MISSING - using defaults]'}")

    # Check Models
    model_path = "models/kalaraksha_damage_classifier.joblib"
    has_model = os.path.exists(model_path)
    print(f"Production ML Model: {'[TRAINED & READY]' if has_model else '[NOT FOUND - run train_classifier.py]'}")

    # Check Sample Assets
    samples = glob.glob("sample_images/*.jpg")
    print(f"Sample Heritage Assets: {len(samples)} pre-packaged assets available.")

    # Check Outputs Directory
    outputs = glob.glob("outputs/*.*")
    print(f"Generated Reports in outputs/: {len(outputs)} files.")

    print("=====================================================")
    print("Quick Run Commands:")
    print("  1. Interactive Web UI:     streamlit run app.py")
    print("  2. Production REST API:    uvicorn api:app --host 0.0.0.0 --port 8000")
    print("  3. Security Verification:  python test_security.py")
    print("  4. Pipeline Test Suite:    python test_pipeline.py")
    print("=====================================================")


def clean_old_outputs(days: int = 7):
    cutoff = time.time() - (days * 86400)
    count = 0
    for p in glob.glob("outputs/*.*"):
        if os.path.basename(p) == ".gitkeep":
            continue
        if os.path.getmtime(p) < cutoff:
            os.remove(p)
            count += 1
    print(f"Cleaned {count} temporary files older than {days} days from outputs/ directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KalaRaksha Log & Health Inspector")
    parser.add_argument("--id", type=str, help="Search for a specific error correlation ID")
    parser.add_argument("--clean-outputs", action="store_true", help="Clean old files from outputs/")
    args = parser.parse_args()

    if args.id:
        inspect_correlation_id(args.id)
    elif args.clean_outputs:
        clean_old_outputs()
    else:
        show_system_health()
