"""
Extract class labels from model metadata, config files, or layer names
"""
import os
import json
import tensorflow as tf
from tensorflow import keras
import h5py

def extract_labels_from_h5(model_path, model_name):
    """Try to extract class labels from .h5 model"""
    print(f"\n{'='*60}")
    print(f"Extracting labels from: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Method 1: Check for custom attributes in HDF5 file
        with h5py.File(model_path, 'r') as f:
            # Check for class_names attribute
            if 'class_names' in f.attrs:
                labels = f.attrs['class_names']
                print(f"[OK] Found class_names in model attributes")
                print(f"Labels: {labels}")
                return list(labels)
            
            # Check for training_config which might have class names
            if 'training_config' in f.attrs:
                config = json.loads(f.attrs['training_config'])
                print(f"[INFO] Training config found, checking for class info...")
        
        # Method 2: Load model and check output layer names
        model = keras.models.load_model(model_path, compile=False)
        output_layer = model.layers[-1]
        
        print(f"[INFO] Output layer: {output_layer.name}")
        print(f"[INFO] Number of classes: {output_layer.output_shape[-1]}")
        
        # Try to get class names from layer config
        if hasattr(output_layer, 'get_config'):
            config = output_layer.get_config()
            if 'class_names' in config:
                print(f"[OK] Found class names in layer config")
                return config['class_names']
        
        # If not found, return placeholder labels
        num_classes = output_layer.output_shape[-1]
        print(f"[WARNING] No class labels found in model metadata")
        print(f"[INFO] Generating placeholder labels for {num_classes} classes")
        return [f"Class_{i}" for i in range(num_classes)]
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None

def extract_labels_from_keras3(model_dir, model_name):
    """Extract labels from Keras 3.0 model directory"""
    print(f"\n{'='*60}")
    print(f"Extracting labels from: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Check config.json
        config_path = os.path.join(model_dir, 'config.json')
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        # Try metadata first
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"[INFO] Metadata: {metadata}")
                if 'class_names' in metadata:
                    print(f"[OK] Found class names in metadata")
                    return metadata['class_names']
        
        # Check config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get output layer info
        if 'config' in config and 'layers' in config['config']:
            layers = config['config']['layers']
            output_layer = layers[-1]
            
            if 'config' in output_layer and 'units' in output_layer['config']:
                num_classes = output_layer['config']['units']
                print(f"[INFO] Number of classes from config: {num_classes}")
                
                # Check if class names are in config
                if 'class_names' in output_layer['config']:
                    return output_layer['config']['class_names']
                
                # Generate placeholders
                print(f"[WARNING] No class labels found")
                return [f"Class_{i}" for i in range(num_classes)]
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None

def try_common_disease_names(crop1, crop2, num_classes):
    """Generate likely disease names based on crop pairs and number of classes"""
    print(f"\n[INFO] Attempting to generate disease names for {crop1} & {crop2}")
    print(f"[INFO] Total classes: {num_classes}")
    
    # Common pattern: Healthy + diseases for each crop
    # Assume roughly equal split: 1 Healthy + (num_classes-1)/2 diseases per crop
    
    labels = ["Healthy"]
    
    diseases_per_crop = (num_classes - 1) // 2
    
    for i in range(diseases_per_crop):
        labels.append(f"{crop1}_Disease_{i+1}")
    
    for i in range(num_classes - len(labels)):
        labels.append(f"{crop2}_Disease_{i+1}")
    
    print(f"[INFO] Generated labels: {labels}")
    return labels

if __name__ == '__main__':
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    all_labels = {}
    
    # Model 1: Rice and Potato
    model1_path = os.path.join(models_dir, 'Model1(Rice and Potato).h5')
    labels = extract_labels_from_h5(model1_path, "Model 1: Rice & Potato")
    if labels:
        all_labels['rice_potato'] = labels
        if labels[0].startswith('Class_'):
            all_labels['rice_potato'] = try_common_disease_names("Rice", "Potato", len(labels))
    
    # Model 2: Corn and Blackgram
    model2_path = os.path.join(models_dir, 'Model2(Corn and Blackgram).h5')
    labels = extract_labels_from_h5(model2_path, "Model 2: Corn & Blackgram")
    if labels:
        all_labels['corn_blackgram'] = labels
        if labels[0].startswith('Class_'):
            all_labels['corn_blackgram'] = try_common_disease_names("Corn", "Blackgram", len(labels))
    
    # Model 4: Wheat and Pumpkin
    model4_dir = os.path.join(models_dir, 'Model4(Wheat and Pumpkin)')
    labels = extract_labels_from_keras3(model4_dir, "Model 4: Wheat & Pumpkin")
    if labels:
        all_labels['wheat_pumpkin'] = labels
        if labels[0].startswith('Class_'):
            all_labels['wheat_pumpkin'] = try_common_disease_names("Wheat", "Pumpkin", len(labels))
    
    # Save to file
    output_file = os.path.join(models_dir, 'class_labels.json')
    with open(output_file, 'w') as f:
        json.dump(all_labels, f, indent=2)
    
    print(f"\n{'='*60}")
    print("CLASS LABELS SUMMARY")
    print(f"{'='*60}")
    for model_name, labels in all_labels.items():
        print(f"\n{model_name}:")
        for i, label in enumerate(labels):
            print(f"  {i}: {label}")
    
    print(f"\n[OK] Labels saved to: {output_file}")
