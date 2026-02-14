# Model Training Folder

This folder contains:
- **Trained models** (.h5, .pth, .keras files)
- **Training notebooks** for retraining models
- **Class labels** (class_labels.json)

## Files

### Models
- `Model1(Rice and Potato).h5` - 12 class disease detection
- `Model2(Corn and Blackgram).h5` - 9 class disease detection (can be retrained)
- `Model3(Tomato and Cotton).pth` - PyTorch model
- `Model4(Wheat and Pumpkin).h5` - Keras 3.0 format

### Training
- `train_model2.ipynb` - Jupyter notebook to retrain Model 2

### Configuration
- `class_labels.json` - Disease class names for all models

## How to Retrain Model 2

1. **Open the notebook:**
   ```bash
   cd backend/models
   jupyter notebook train_model2.ipynb
   ```

2. **Prepare your dataset:**
   - Create folder: `datasets/corn_blackgram/train/`
   - Organize by class (e.g., `Blackgram_Healthy/`, `Corn_Rust/`, etc.)

3. **Run all cells** in the notebook

4. **Update class_labels.json** with new classes if changed

5. **Test predictions** with the multi-model system!

---

**Note:** Large model files (.h5, .pth) are gitignored to keep the repo size small.
