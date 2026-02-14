# Agridetect AI �

Agridetect AI is an AI-powered crop health assistant that detects plant diseases from images, provides actionable recommendations, and helps farmers track scans, alerts, and history. It combines local ML inference with a conversational assistant for guidance.

This project is part of my learning journey in **full-stack development** and **real-world project deployment**.

---

## 🚀 Highlights
 
- Image-based disease diagnosis with local ML models
- Location and weather-aware tips and insights
- Multilingual UI (English and Indian languages)
- Voice input/output for accessibility
- History, alerts, and community views

---

## 📌 Key Features
 
- Upload and scan crop images for instant results
- Dashboard with recent scans and alerts
- Chat assistant backed by Gemini for agronomy guidance
- Privacy-conscious: models and venv stay out of Git

---

## 🛠️ Tech Stack

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS

### Backend
- Flask (Python) API
- SQLite database (agridetect.db)
- Local ML models (.h5/.pt placed under backend/models)
- Gemini-powered chat assistant

### Tools & Deployment
- Git & GitHub
- Vercel (deployment)
- npm
 
---
 
## ⚙️ Setup
 
### Backend
- Create backend/.env and set GEMINI_API_KEY
- Place model files under backend/models (not stored in Git)
- Install deps: `pip install -r backend/requirements.txt`
- Run: `python backend/app.py`
 
### Frontend
- Install deps: `npm install`
- Run dev: `npm run dev`
 
### Repo Policy
- Models and virtual environments are not committed
- See .gitignore and backend/.gitignore for excluded paths
 
---
 
---

## 📂 Project Structure

```text
Agridetect-AI/
│
├── public/                # Static assets
├── src/                   # React source code
│   ├── components/        # Reusable UI components
│   ├── pages/             # Application pages
│   └── main.tsx           # Entry point
│
├── backend/               # Flask API + models (local only)
├── .gitignore             # Git ignored files
├── package.json           # Project dependencies
├── vite.config.ts         # Vite configuration
└── README.md              # Project documentation
