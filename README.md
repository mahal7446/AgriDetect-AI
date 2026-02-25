# 🌱 AgriDetect AI

**AgriDetect AI** is a full-stack agricultural management and disease detection platform. It combines Deep Learning models with Google's Generative AI to give farmers real-time crop disease diagnoses, localized community alerts, predictive analytics, and expert agricultural advice — all from a single dashboard.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| **🔍 AI Disease Detection** | Upload or capture crop photos to detect diseases across **8 crops** (Rice, Potato, Corn, Blackgram, Tomato, Cotton, Wheat, Pumpkin) using EfficientNet-B3 and PyTorch models. |
| **📍 Community Alerts** | District-based disease alert system — you only see alerts relevant to your location. Submit and manage alerts with image evidence. |
| **💬 Agri-Chatbot** | Context-aware AI chatbot powered by Google Gemini. It understands your scan history to provide tailored crop recommendations. |
| **📊 Predictive Analytics** | Visualize farm health trends, yield forecasts, disease distribution, and risk assessments through interactive charts. |
| **🌐 Multi-Language Support** | Built-in i18n support for multiple languages via i18next. |
| **🔄 Scan History** | Track every scan, monitor disease progress, and manage your agricultural data over time. |
| **🌤️ Weather Integration** | Real-time weather data via OpenWeather API, displayed on the dashboard. |
| **🔐 Auth & Email Verification** | Secure signup/login with bcrypt hashing, email verification via SMTP, and OTP support. |

---

## 🛠️ Tech Stack

### Frontend

| Layer | Technology |
|---|---|
| Framework | [React 18](https://reactjs.org/) + [Vite 5](https://vitejs.dev/) |
| Language | [TypeScript 5](https://www.typescriptlang.org/) |
| Styling | [Tailwind CSS 3](https://tailwindcss.com/) + [Shadcn UI](https://ui.shadcn.com/) (Radix primitives) |
| State / Data | [React Context](https://react.dev/learn/passing-data-deeply-with-context) + [TanStack React Query](https://tanstack.com/query) |
| Routing | [React Router 6](https://reactrouter.com/) |
| Charts | [Recharts](https://recharts.org/) |
| 3D | [React Three Fiber](https://docs.pmnd.rs/react-three-fiber) + [Three.js](https://threejs.org/) |
| i18n | [i18next](https://www.i18next.com/) |
| Testing | [Vitest](https://vitest.dev/) + [Testing Library](https://testing-library.com/) |

### Backend

| Layer | Technology |
|---|---|
| Framework | [Flask 3](https://flask.palletsprojects.com/) (Python) |
| Database | [SQLite3](https://www.sqlite.org/) |
| ML / Deep Learning | [TensorFlow / Keras](https://www.tensorflow.org/) + [PyTorch](https://pytorch.org/) |
| Generative AI | [Google Gemini API](https://ai.google.dev/) |
| Auth | [bcrypt](https://pypi.org/project/bcrypt/) + [itsdangerous](https://itsdangerous.palletsprojects.com/) |
| Production Server | [Gunicorn](https://gunicorn.org/) |

---

## 📂 Project Structure

```
AgriDetect-AI/
├── README.md
├── .gitignore
│
├── backend/                        # Python / Flask API
│   ├── app.py                      # Main Flask entry point (31 API routes)
│   ├── database.py                 # SQLite schema & queries
│   ├── chat_service.py             # Google Gemini chatbot integration
│   ├── working_model_manager.py    # ML model loading & prediction
│   ├── email_service.py            # SMTP email verification
│   ├── location_helpers.py         # District extraction & normalization
│   ├── append_db_functions.py      # Location-based alert queries
│   ├── requirements.txt            # Python dependencies
│   ├── .env.example                # Backend env template
│   ├── start_backend.bat           # Windows startup script
│   ├── start_backend.sh            # Linux/macOS startup script
│   ├── models/                     # Trained ML model files
│   │   ├── crop_classifier_new.h5
│   │   ├── Model1(Rice and Potato).h5
│   │   ├── Model2(Corn and Blackgram).h5
│   │   ├── Model3(Tomato and Cotton).pt
│   │   ├── Model4(Wheat and Pumpkin).h5
│   │   ├── class_labels.json
│   │   └── class_names.json
│   └── uploads/                    # User-uploaded images
│       ├── scan_images/
│       ├── alert_images/
│       └── profile_pictures/
│
└── frontend/                       # React / Vite SPA
    ├── index.html                  # HTML entry point
    ├── package.json                # NPM dependencies & scripts
    ├── vite.config.ts              # Vite configuration
    ├── tsconfig.json               # TypeScript configuration
    ├── tailwind.config.ts          # Tailwind CSS theme & plugins
    ├── components.json             # Shadcn UI configuration
    ├── vitest.config.ts            # Test configuration
    ├── eslint.config.js            # Linting rules
    ├── postcss.config.js           # PostCSS plugins
    ├── vercel.json                 # Vercel deployment config
    ├── .env.example                # Frontend env template
    ├── public/                     # Static assets (favicon, images)
    └── src/
        ├── main.tsx                # App bootstrap
        ├── App.tsx                 # Root component & routes
        ├── pages/                  # Page-level views
        │   ├── LandingPage.tsx
        │   ├── DashboardPage.tsx
        │   ├── UploadPage.tsx
        │   ├── ResultPage.tsx
        │   ├── HistoryPage.tsx
        │   ├── AnalyticsPage.tsx
        │   ├── CommunityPage.tsx
        │   ├── ProfilePage.tsx
        │   └── auth/               # Login, Signup, Verify
        ├── components/             # Reusable UI components
        │   ├── ui/                 # Shadcn / Radix primitives
        │   ├── dashboard/          # Dashboard widgets
        │   ├── landing/            # Landing page sections
        │   ├── layout/             # Navbar, Layout wrapper
        │   ├── chat/               # Chatbot FAB
        │   └── profile/            # Profile components
        ├── contexts/               # React Context providers
        ├── hooks/                  # Custom hooks (toast, geolocation, voice, weather)
        ├── lib/                    # API client & utilities
        ├── i18n/                   # Internationalization config & locale files
        ├── data/                   # Static data (Indian cities list)
        ├── types/                  # TypeScript type definitions
        └── test/                   # Test setup & specs
```

---

## 🏁 Getting Started

### Prerequisites

| Tool | Version |
|---|---|
| Node.js | v18 or higher |
| Python | 3.9 or higher |
| pip | latest |

### 1. Clone the Repository

```bash
git clone https://github.com/mahal7446/AgriDetect-AI.git
cd AgriDetect-AI
```

### 2. Backend Setup

```bash
cd backend

# (Optional) Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env from template and fill in your keys
cp .env.example .env

# Start the server (runs on http://localhost:5000)
python app.py
```

> **Note:** The SQLite database (`agridetect.db`) and `uploads/` directories are **created automatically** at runtime — no manual setup needed.

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env from template and fill in your keys
cp .env.example .env

# Start the dev server (runs on http://localhost:8081)
npm run dev
```

---

## ⚙️ Environment Variables

### Backend — `backend/.env`

| Variable | Required | Description | Default |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key | — |
| `SECRET_KEY` | Yes | App secret for token signing | — |
| `SMTP_SERVER` | No | SMTP server for email verification | `smtp.gmail.com` |
| `SMTP_PORT` | No | SMTP port | `587` |
| `SMTP_EMAIL` | No | Sender email address | — |
| `SMTP_PASSWORD` | No | SMTP app password | — |
| `OPENWEATHER_API_KEY` | No | OpenWeather API key | — |
| `FRONTEND_URL` | No | Frontend origin (for CORS / emails) | `http://localhost:8081` |
| `DATABASE_URL` | No | SQLite database path | `sqlite:///agri_trail.db` |

### Frontend — `frontend/.env`

| Variable | Required | Description | Default |
|---|---|---|---|
| `VITE_API_URL` | Yes | Backend API base URL | `http://localhost:5000` |
| `VITE_OPENWEATHER_API_KEY` | No | OpenWeather API key (weather widget) | — |
| `VITE_GEMINI_API_KEY` | No | Gemini API key (client-side chatbot) | — |

---

## 📜 Available Scripts

### Frontend (`cd frontend`)

| Command | Description |
|---|---|
| `npm run dev` | Start Vite dev server with HMR |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Preview production build locally |
| `npm run lint` | Run ESLint |
| `npm run test` | Run tests with Vitest |
| `npm run test:watch` | Run tests in watch mode |

### Backend (`cd backend`)

| Command | Description |
|---|---|
| `python app.py` | Start Flask dev server on port 5000 |
| `./start_backend.sh` | Linux/macOS startup script |
| `start_backend.bat` | Windows startup script |

---

## 🧠 ML Models

The platform uses **5 trained models** to classify diseases across 8 crops:

| Model | File | Crops | Framework |
|---|---|---|---|
| Crop Classifier | `crop_classifier_new.h5` | General crop identification | TensorFlow / Keras |
| Model 1 | `Model1(Rice and Potato).h5` | Rice, Potato | TensorFlow / Keras |
| Model 2 | `Model2(Corn and Blackgram).h5` | Corn, Blackgram | TensorFlow / Keras |
| Model 3 | `Model3(Tomato and Cotton).pt` | Tomato, Cotton | PyTorch |
| Model 4 | `Model4(Wheat and Pumpkin).h5` | Wheat, Pumpkin | TensorFlow / Keras |

> **Note:** Model files (`.h5`, `.pt`) are excluded from Git via `.gitignore` due to their large size. See `backend/models/README.md` for download instructions.

---

## 🌐 API Endpoints

<details>
<summary>Click to expand all 31 routes</summary>

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `POST` | `/predict` | Upload image for disease prediction |
| `POST` | `/api/auth/signup` | Register new user |
| `GET` | `/api/auth/verify-email/<token>` | Verify email address |
| `POST` | `/api/auth/resend-verification` | Resend verification email |
| `POST` | `/api/auth/login` | User login |
| `GET` | `/api/profile/get` | Get user profile |
| `POST` | `/api/profile/update` | Update user profile |
| `POST` | `/api/profile/upload-picture` | Upload profile picture |
| `GET` | `/uploads/profile_pictures/<file>` | Serve profile picture |
| `GET` | `/api/profile/stats` | Get user stats |
| `GET` | `/api/profile/accuracy` | Get prediction accuracy |
| `GET` | `/api/stats/total-users` | Get total user count |
| `GET` | `/api/alerts/recent` | Get recent disease alerts |
| `POST` | `/api/alerts/submit` | Submit a new alert |
| `GET` | `/uploads/alert_images/<file>` | Serve alert image |
| `GET` | `/api/alerts/by-location` | Get alerts by user location |
| `DELETE` | `/api/alerts/delete/<id>` | Delete an alert |
| `POST` | `/api/alerts/update/<id>` | Update an alert |
| `GET` | `/api/alerts/new-count` | Get new alerts count |
| `GET` | `/api/profile/notification-preference` | Get notification settings |
| `POST` | `/api/profile/update-notification-preference` | Update notifications |
| `POST` | `/api/history/save` | Save a scan to history |
| `GET` | `/api/history/get` | Get scan history |
| `DELETE` | `/api/history/delete/<id>` | Delete a scan |
| `GET` | `/uploads/scan_images/<file>` | Serve scan image |
| `GET` | `/api/analytics/summary` | Get analytics summary |
| `GET` | `/api/analytics/charts` | Get chart data |
| `GET` | `/api/analytics/reports` | Get analytics reports |
| `POST` | `/api/chat` | Send message to AI chatbot |
| `POST` | `/api/chat/greeting` | Get chatbot greeting |

</details>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## �️ Git & Repository Notes

The following are **gitignored** and never committed:

- `agridetect.db` — SQLite database (auto-created at runtime)
- `backend/uploads/` — User-uploaded images (auto-created at runtime)
- `*.log` / `server_output*.txt` — Server logs and debug output
- `.env` files — Sensitive API keys (use `.env.example` as a template)
- `node_modules/` / `dist/` — Build artifacts
- `backend/models/*.h5` / `*.pt` — Large ML model weights

---

## �📄 License

This project is licensed under the MIT License.
