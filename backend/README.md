# Backend API Server

This Flask server handles ML model predictions for plant disease detection.

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your ML model:**
   - Create a `models` directory: `mkdir models`
   - Place your `.h5` model file in `backend/models/model.h5`

3. **Update model configuration:**
   - Edit `app.py` and update:
     - `preprocess_image()` function: Adjust image resize dimensions (currently 224x224)
     - `get_class_labels()` function: Update with your model's class labels
     - Crop detection logic if needed

4. **Run the server:**
   ```bash
   python app.py
   ```
   
   The server will run on `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/health`
- Returns server and model status

### Prediction
- **POST** `/predict`
- **Body:** Form-data with `image` file
- **Response:** JSON with prediction results

## Model Requirements

- Model file should be in `.h5` format (Keras/TensorFlow)
- Default input size: 224x224 (update in `preprocess_image()` if different)
- Model should output class probabilities
