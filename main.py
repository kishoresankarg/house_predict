from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import joblib
import os
import sys

app = FastAPI()

# Load model if available; if loading fails, keep model = None and return a helpful error on predict
model = None
try:
    model = joblib.load("house_model.pkl")
except Exception as e:
    # Don't raise here — allow the app to start so the deployment can show logs and diagnostics.
    print(f"Warning: failed to load model: {e}")


class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    html = """
    <html>
      <head>
        <title>House Price Predictor</title>
        <meta charset="utf-8" />
      </head>
      <body>
        <h1>House Price Prediction API</h1>
        <p>Use the <a href="/docs">OpenAPI docs</a> to try the <code>/predict</code> endpoint.</p>
        <p>POST JSON to <code>/predict</code> with body <code>{"data": [features...]}</code>.</p>
        <pre>Example features: [8.3252,41.0,6.98,1.02,322,2.55,37.88,-122.23]</pre>
      </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)


@app.post("/predict")
def predict(input: Input = Input()):
    if model is None:
        return JSONResponse(status_code=500, content={"detail": "Model not available on server. Check logs."})
    try:
        pred = model.predict([input.data])
        return {"prediction": float(pred[0])}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Prediction failed: {e}"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=port)
    except OSError as e:
        print(f"ERROR: Failed to bind to 0.0.0.0:{port} — {e}")
        print("Try setting a different PORT environment variable, e.g. $env:PORT=9000")
        sys.exit(1)