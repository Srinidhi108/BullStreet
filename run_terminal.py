"""
run_terminal.py - Master Launcher for BullStreet Bloomberg Terminal
Starts the FastAPI backend and serves the Bloomberg Terminal interface.
"""
import os
import sys
import webbrowser
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BullStreetLauncher")

BANNER = """
================================================================================
  BULLSTREET INTELLIGENCE TERMINAL (BST)
  Dual ML Prediction, Multi-Horizon Trajectory & Risk Engine
================================================================================
  Host:   http://127.0.0.1:8000
  Engine: Dual Classification + Dual Regression + Local SHAP XAI
================================================================================
"""

def main():
    print(BANNER)
    # Ensure artifacts directory exists and model is trained
    bundle_path = os.path.join("artifacts", "model_bundle.joblib")
    if not os.path.exists(bundle_path):
        print("[!] Model bundle not found. Training model engine...")
        from src.model_engine import engine
        engine.train_and_evaluate()
        print("[+] Model training complete.")

    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    print(f"\n[*] Launching Bloomberg Terminal Interface at: {url}")
    print("[*] Press Ctrl+C to terminate the terminal.\n")

    # Open browser automatically if not in a headless test
    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("src.api:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
