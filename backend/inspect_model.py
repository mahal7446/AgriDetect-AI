"""
Helper script to inspect your ML model and determine its configuration
Run this to see your model's input/output shapes and number of classes
"""
import os
import tensorflow as tf
from tensorflow import keras

def inspect_model():
    """Inspect the model file to get its configuration"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    possible_names = [
        'efficientnet_b3_final (1).h5',
        'model.h5',
        'efficientnet_b3_final.h5',
    ]
    
    model_path = None
    for name in possible_names:
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("[ERROR] Model file not found!")
        print(f"Looking in: {models_dir}")
        return
    
    try:
        print(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        
        # Input shape
        input_shape = model.input_shape
        print(f"\n[OK] Input Shape: {input_shape}")
        if len(input_shape) == 4:  # (batch, height, width, channels)
            height, width = input_shape[1], input_shape[2]
            print(f"   -> Image size should be: {height}x{width}")
        
        # Output shape
        output_shape = model.output_shape
        print(f"\n[OK] Output Shape: {output_shape}")
        if len(output_shape) == 2:  # (batch, num_classes)
            num_classes = output_shape[1]
            print(f"   -> Number of classes: {num_classes}")
        
        # Model summary
        print(f"\nModel Summary:")
        print("-"*60)
        model.summary()
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print(f"\n1. Update image resize in app.py to: ({height}, {width})")
        print(f"2. Update get_class_labels() in app.py with your {num_classes} class names")
        print(f"3. Make sure class labels are in the SAME ORDER as your model's output")
        
    except Exception as e:
        print(f"\n[ERROR] Error loading model: {str(e)}")
        print("\nPossible issues:")
        print("- TensorFlow/Keras version mismatch")
        print("- Model file corrupted")
        print("- Missing dependencies")

if __name__ == '__main__':
    inspect_model()
