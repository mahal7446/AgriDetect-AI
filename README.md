# 🌱 AgriDetect AI

**AgriDetect AI** is a state-of-the-art agricultural management and disease detection platform. It leverages Deep Learning models and Generative AI to empower farmers with real-time insights, disease diagnoses, and expert agricultural advice.

---

## 🚀 Key Features

*   **🔍 AI Disease Detection**: Upload or take photos of crops to detect diseases with high confidence using advanced ML models (EfficientNet-B3).
*   **📍 Local Community Alerts**: Stay informed with localized disease alerts. The platform uses strict district-based isolation, so you only see alerts relevant to your specific location.
*   **💬 Agri-Chatbot**: Get context-aware agricultural advice from our AI chatbot, powered by Google's Gemini. It understands your detection history to provide tailored recommendations.
*   **📊 Predictive Analytics**: Visualize your farm's health trends, yield forecasts, and risk assessments through an intuitive analytics dashboard.
*   **📱 Multi-Language Support**: Accessible to a global audience with built-in support for multiple languages.
*   **🔄 Real-time History**: Track every scan, monitor disease progress, and manage your agricultural data in one place.

---

## 🛠️ Tech Stack

### Frontend
- **Framework**: [React](https://reactjs.org/) + [Vite](https://vitejs.dev/)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) + [Shadcn UI](https://ui.shadcn.com/)
- **Charts**: [Recharts](https://recharts.org/)
- **Internationalization**: [i18next](https://www.i18next.com/)

### Backend
- **Framework**: [Flask](https://flask.palletsprojects.com/) (Python)
- **Database**: [SQLite3](https://www.sqlite.org/)
- **ML/AI**: [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [PyTorch](https://pytorch.org/)
- **Generative AI**: [Google Gemini API](https://ai.google.dev/)

---

## 📂 Project Structure

```text
AgriDetect-AI/
├── backend/                # Flask server, ML models, and Database logic
│   ├── models/             # Trained (.h5 / .pth) model files
│   ├── uploads/            # User-uploaded images (scans, alerts)
│   ├── app.py              # Main Flask entry point
│   ├── database.py         # SQLite connection and queries
│   └── chat_service.py     # Gemini AI integration
├── src/                    # React Frontend source code
│   ├── components/         # Reusable UI elements
│   ├── pages/              # Main application views
│   ├── contexts/           # React state management (Auth, Notifications)
│   └── lib/                # API client and utilities
├── public/                 # Static assets
└── package.json            # Frontend dependencies and scripts
```

---

## 🏁 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.9+)

### 1. Backend Setup
```bash
# Navigate to the project root
cd AgriDetect-AI

# Install Python dependencies
pip install -r backend/requirements.txt

# Create .env based on .env.example and add your API keys
# Required: GEMINI_API_KEY
python backend/app.py
```

### 2. Frontend Setup
```bash
# Install NPM dependencies
npm install

# Start the development server
npm run dev
```

---

## ⚙️ Configuration
Create a `.env` file in the root and backend directories with the following:

**Backend (`backend/.env`):**
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Frontend (`.env`):**
```env
VITE_API_URL=http://localhost:5000

VITE_OPENWEATHER_API_KEY=your_openweather_api_key_here
```

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.

## 📄 License
This project is licensed under the MIT License.
