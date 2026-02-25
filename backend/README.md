# Backend — AgriDetect AI API Server

Flask API server powering AgriDetect AI with ML-based plant disease detection, user auth, community alerts, analytics, and an AI chatbot.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy env template and add your API keys
cp .env.example .env

# Run the server (http://localhost:5000)
python app.py
```

> The SQLite database and `uploads/` directories are created automatically on first run.

## Key Modules

| File | Purpose |
|---|---|
| `app.py` | Flask entry point — 31 API routes |
| `database.py` | SQLite schema, queries, and migrations |
| `working_model_manager.py` | Multi-model loading and prediction pipeline |
| `chat_service.py` | Google Gemini chatbot integration |
| `email_service.py` | SMTP email verification & OTP |
| `location_helpers.py` | District extraction & normalization |
| `verification_tokens.py` | Secure token generation for email verification |

## ML Models

Place trained model files in the `models/` directory:

| File | Crops | Framework |
|---|---|---|
| `crop_classifier_new.h5` | General crop identification | TensorFlow / Keras |
| `Model1(Rice and Potato).h5` | Rice, Potato | TensorFlow / Keras |
| `Model2(Corn and Blackgram).h5` | Corn, Blackgram | TensorFlow / Keras |
| `Model3(Tomato and Cotton).pt` | Tomato, Cotton | PyTorch |
| `Model4(Wheat and Pumpkin).h5` | Wheat, Pumpkin | TensorFlow / Keras |

> Model weights are excluded from Git due to their large size. See `models/README.md` for download instructions.

## Environment Variables

See `.env.example` for the full list. Key variables:

- `GEMINI_API_KEY` — Google Gemini API key (required for chatbot)
- `SECRET_KEY` — App secret for token signing
- `SMTP_*` — Email service configuration
- `OPENWEATHER_API_KEY` — Weather data integration
