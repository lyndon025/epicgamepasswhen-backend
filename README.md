# Epic & Game Pass Predictor - Backend

Machine learning backend for predicting when games will arrive on subscription services.

## Features

-   **Tiered Prediction**: Uses historical data, XGBoost ML, and first-party checks.
-   **Service-Oriented Architecture**: Logic separated into `services/` module.
-   **Deployment Ready**: Configured for Render (Default) and Fly.io.

## Local Development

1.  **Create venv**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run**:
    ```bash
    python app.py
    ```
    Runs on `http://localhost:5000`.

## Deployment

### Render (Default)
This project is optimized for Render.
1.  Connect your GitHub repository to Render.
2.  Select **Web Service**.
3.  Render will automatically detect Python/Flask.
    -   **Build Command**: `pip install -r requirements.txt`
    -   **Start Command**: `gunicorn app:app` (or `python app.py`)
4.  **Environment Variables**:
    -   `PYTHON_VERSION`: `3.11.0` (optional but recommended)

### Fly.io
The project also includes `fly.toml` and `Dockerfile` for Fly.io.

1.  **Install flyctl**: [https://fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/)
2.  **Launch**:
    ```bash
    fly launch
    ```
3.  **Deploy**:
    ```bash
    fly deploy
    ```

## API

-   `POST /api/predict`: Get game prediction.
    -   Body: `{ "game_name": "...", "platform": "epic|gamepass|psplus", ... }`
-   `GET /api/health`: Health check.
