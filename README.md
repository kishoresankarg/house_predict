# House Price Predictor

This repo contains a small FastAPI app that serves a pre-trained model for house price prediction.

Quick notes for deployment on Render (or similar platforms):

- Ensure `house_model.pkl` is present in the deployed repository or otherwise available to the service at runtime.
- The server expects to be started with a command like:

  web: uvicorn main:app --host 0.0.0.0 --port $PORT

  (A `Procfile` is included with that command.)

- If you see a browser show `{"detail": "Not Found"}`, it usually means the running process isn't serving the expected `main:app` (check your start command) or the deployed code doesn't include the root route — redeploy after committing changes.

Troubleshooting
- Check service logs in Render (or your hosting provider) for startup errors and model-loading warnings.
- Common issues:
  - Port binding errors: ensure the service uses the `$PORT` env variable Render provides.
  - Model loading errors: ensure `house_model.pkl` exists and the scikit-learn version is compatible.

To test locally (PowerShell):

```powershell
$env:PORT = 9000; python main.py
```

Then open `http://localhost:9000/` or `http://localhost:9000/docs`.
