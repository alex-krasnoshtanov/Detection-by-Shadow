"""A single-process web demo for the trained model.

Optional: needs the ``demo`` extra (``pip install -e ".[demo]"``). Nothing in
the core package imports from here, so the FastAPI dependency stays optional.

    uvicorn shadow_detection.demo.app:app --port 8000
"""
