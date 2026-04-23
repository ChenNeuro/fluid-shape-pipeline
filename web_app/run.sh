#!/usr/bin/env python
"""Wake-Field Shape Classifier — Interactive Web App"""
from app import app, ROOT, MODEL_PATH

print("=" * 60)
print("  Wake-Field Shape Classifier")
print("=" * 60)
print(f"  Project root : {ROOT}")
print(f"  Model       : {MODEL_PATH}")
print(f"  Model exists: {MODEL_PATH.exists()}")
print()
print("  Open: http://127.0.0.1:5000")
print("  Press Ctrl+C to stop")
print("=" * 60)
app.run(host="127.0.0.1", port=5000, debug=False)
