🌿 AgriDetect-AI

AgriDetect-AI is an AI-powered plant disease detection web application designed to help farmers and agricultural enthusiasts identify crop diseases from leaf images. The system uses Machine Learning models to analyze uploaded images and provides disease predictions along with helpful recommendations.

🚀 Features

📸 Upload plant leaf images for disease detection

🤖 AI-based disease prediction using trained ML models

📊 Dashboard with scan results

💬 AI Chat Assistant for agriculture guidance

🌦️ Weather-aware recommendations (if configured)

🗂️ Scan history tracking

🔐 Secure backend with environment variable configuration

🌐 Modern and responsive UI

🏗️ Tech Stack
Frontend

React 18

TypeScript

Vite

Tailwind CSS

Backend

Python

Flask

SQLite Database

TensorFlow / Keras (.h5 models)

Gemini API (for chatbot)

Deployment

GitHub

Render / Vercel (optional deployment platforms)

📂 Project Structure

AgriDetect-AI/

backend/
    models/ # Trained ML model files (.h5)
    app.py # Main Flask application
    chat_service.py # Chat assistant logic
    database.py # Database initialization
    requirements.txt # Python dependencies

public/ # Static frontend assets

src/
    components/ # Reusable React components
    pages/ # Application pages
    App.tsx # Main React App
    main.tsx # Entry point

.env.example # Environment variables template
package.json # Node dependencies
tailwind.config.ts # Tailwind configuration
vite.config.ts # Vite configuration
README.md

⚙️ Installation & Setup
1️⃣ Clone the Repository

git clone https://github.com/mahal7446/AgriDetect-AI.git

cd AgriDetect-AI

🐍 Backend Setup
Step 1: Navigate to backend

cd backend

Step 2: Create virtual environment (recommended)

python -m venv .venv

For Windows:
.venv\Scripts\activate

For Linux/Mac:
source .venv/bin/activate

Step 3: Install dependencies

pip install -r requirements.txt

Step 4: Setup environment variables

Create a .env file inside the backend folder and add:

GEMINI_API_KEY=your_api_key_here

Step 5: Add ML Models

Place your trained .h5 model files inside:

backend/models/

Step 6: Run backend server

python app.py

Backend runs on:
http://localhost:5000

⚛️ Frontend Setup
Step 1: Go to root directory

cd ..

Step 2: Install dependencies

npm install

Step 3: Run development server

npm run dev

Frontend runs on:
http://localhost:5173

🧪 How to Use

Start backend server.

Start frontend server.

Open the frontend in your browser.

Upload a plant leaf image.

View prediction results.

Use the AI chatbot for further assistance.

🔐 Environment Variables

GEMINI_API_KEY – API key for chatbot integration

📦 Requirements

Python 3.9+

Node.js 18+

pip

npm

🚀 Deployment (Optional)

Frontend:

Vercel

Netlify

Render

Backend:

Render

Railway

Any VPS server

Make sure environment variables are configured properly during deployment.

🤝 Contributing

Fork the repository

Create a new branch

Make changes

Submit a Pull Request

📜 License

This project currently does not include a license file.
You may consider adding an MIT License for open-source usage.

👨‍💻 Author

Mahaling S M
GitHub: https://github.com/mahal7446

🌱 Future Improvements

Multi-language support

More crop models

Real-time weather API integration

Mobile-friendly PWA version

Cloud model hosting
