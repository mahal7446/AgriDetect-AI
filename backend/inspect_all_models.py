"""
Inspect multiple ML models to determine their configurations
"""
import os
import tensorflow as tf
from tensorflow import keras
import json

def inspect_h5_model(model_path, model_name):
    """Inspect .h5 model"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {model_name}")
    print(f"{'='*60}")
    
    try:
        model = keras.models.load_model(model_path, compile=False)
        
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        height, width = input_shape[1], input_shape[2] if len(input_shape) == 4 else (None, None)
        num_classes = output_shape[1] if len(output_shape) == 2 else None
        
        print(f"[OK] Input Shape: {input_shape} -> Image size: {height}x{width}")
        print(f"[OK] Output Shape: {output_shape} -> Classes: {num_classes}")
        print(f"[OK] Total Layers: {len(model.layers)}")
        
        return {
            'format': '.h5',
            'input_shape': input_shape,
            'image_size': (height, width),
            'num_classes': num_classes,
            'output_shape': output_shape
        }
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None

def inspect_keras3_model(model_dir, model_name):
    """Inspect Keras 3.0 model (config.json + weights.h5)"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {model_name}")
    print(f"{'='*60}")
    
    try:
        config_path = os.path.join(model_dir, 'config.json')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"[OK] Model Type: Keras 3.0 format")
        print(f"[OK] Config file found")
        
        # Try to determine shapes from config
        if 'config' in config and 'layers' in config['config']:
            layers = config['config']['layers']
            input_layer = layers[0]
            output_layer = layers[-1]
            
            print(f"[OK] Input layer: {input_layer.get('class_name', 'Unknown')}")
            print(f"[OK] Output layer: {output_layer.get('class_name', 'Unknown')}")
            
            # Try to get batch_shape from input
            if 'config' in input_layer and 'batch_shape' in input_layer['config']:
                batch_shape = input_layer['config']['batch_shape']
                print(f"[OK] Input shape: {batch_shape}")
                if len(batch_shape) == 4:
                    height, width = batch_shape[1], batch_shape[2]
                    print(f"  -> Image size: {height}x{width}")
        
        return {
            'format': 'keras3',
            'config': config
        }
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None

def inspect_pytorch_model(model_path, model_name):
    """Inspect PyTorch .pth model"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {model_name}")
    print(f"{'='*60}")
    
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check if it's state_dict or full model
        if isinstance(checkpoint, dict):
            print(f"[OK] Type: State Dictionary")
            print(f"[OK] Keys: {list(checkpoint.keys())[:5]}...")
            
            # Count parameters
            num_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
            print(f"[OK] Total parameters: {num_params:,}")
        else:
            print(f"[OK] Type: Full Model")
            print(f"[OK] Model class: {type(checkpoint)}")
        
        return {
            'format': '.pth',
            'checkpoint_type': 'state_dict' if isinstance(checkpoint, dict) else 'full_model'
        }
    except ImportError:
        print(f"[ERROR] PyTorch not installed. Run: pip install torch torchvision")
        return None
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None

if __name__ == '__main__':
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    print("\n" + "="*60)
    print("MULTI-MODEL INSPECTION")
    print("="*60)
    
    results = {}
    
    # Model 1: Rice and Potato (.h5)
    model1_path = os.path.join(models_dir, 'Model1(Rice and Potato).h5')
    if os.path.exists(model1_path):
        results['model1'] = inspect_h5_model(model1_path, "Model 1: Rice & Potato")
    
    # Model 2: Corn and Blackgram (.h5)
    model2_path = os.path.join(models_dir, 'Model2(Corn and Blackgram).h5')
    if os.path.exists(model2_path):
        results['model2'] = inspect_h5_model(model2_path, "Model 2: Corn & Blackgram")
    
    # Model 3: Tomato and Cotton (.pth)
    model3_path = os.path.join(models_dir, 'Model3(Tomato and Cotton).pth')
    if os.path.exists(model3_path):
        results['model3'] = inspect_pytorch_model(model3_path, "Model 3: Tomato & Cotton")
    
    # Model 4: Wheat and Pumpkin (Keras 3.0)
    model4_dir = os.path.join(models_dir, 'Model4(Wheat and Pumpkin)')
    if os.path.exists(model4_dir):
        results['model4'] = inspect_keras3_model(model4_dir, "Model 4: Wheat & Pumpkin")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nFound {len([r for r in results.values() if r is not None])} valid models")
    
    for model_name, info in results.items():
        if info:
            print(f"\n[OK] {model_name}: {info['format']}")
