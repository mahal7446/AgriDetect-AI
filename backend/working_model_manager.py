# Fixed Working Model Manager - Clean Implementation

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F

class WorkingModelManager:
    def __init__(self):
        """Clean, working model manager"""
        self.models = {}
        self.model_types = {}
        self.input_sizes = {}
        self.confidence_threshold = 0.85
        self.labels = {}
        
        try:
            base_dir = os.path.dirname(__file__)
            models_dir = os.path.join(base_dir, 'models')
            entries = os.listdir(models_dir)
            for fname in entries:
                fpath = os.path.join(models_dir, fname)
                name = fname.lower()
                is_dir = os.path.isdir(fpath)
                is_model_file = name.endswith('.h5') or name.endswith('.keras') or name.endswith('.pth') or name.endswith('.pt')
                if not (is_model_file or is_dir):
                    continue
                if 'crop_classifier' in name:
                    continue
                key = fname
                if '(' in fname and ')' in fname:
                    inside = fname[fname.find('(')+1:fname.find(')')]
                    norm = inside.lower()
                    norm = norm.replace('&', 'and')
                    norm = norm.replace(' and ', '_').replace(' ', '_')
                    key = norm
                else:
                    norm = name.replace('&', 'and')
                    key = norm.replace('.h5', '').replace('.keras', '').replace('.pth', '')
                try:
                    if is_dir:
                        try:
                            model = keras.models.load_model(fpath, compile=False)
                            self.models[key] = model
                            self.model_types[key] = 'keras'
                            print(f"[OK] Loaded directory model {key}")
                        except Exception as e_dir:
                            print(f"[WARNING] Could not load directory {fname}: {str(e_dir)}")
                            continue
                    elif name.endswith('.pt'):
                        try:
                            model = torch.jit.load(fpath, map_location='cpu')
                            model.eval()
                            self.models[key] = model
                            self.model_types[key] = 'torch'
                        except Exception as e_pt:
                            print(f"[WARNING] Could not load TorchScript {fname}: {str(e_pt)}")
                            continue
                    elif name.endswith('.pth'):
                        try:
                            model = torch.jit.load(fpath, map_location='cpu')
                            model.eval()
                            self.models[key] = model
                            self.model_types[key] = 'torch'
                        except Exception:
                            obj = torch.load(fpath, map_location='cpu')
                            if hasattr(obj, 'eval'):
                                obj.eval()
                                self.models[key] = obj
                                self.model_types[key] = 'torch'
                            else:
                                print(f"[WARNING] {fname} appears to be a state_dict; architecture is required to load it")
                                continue
                    else:
                        try:
                            self.models[key] = keras.models.load_model(fpath, compile=False)
                        except Exception as e_load:
                            try:
                                from tensorflow.keras import layers as KL
                                class PatchedDense(KL.Dense):
                                    @classmethod
                                    def from_config(cls, config):
                                        cfg = dict(config)
                                        cfg.pop('quantization_config', None)
                                        return super().from_config(cfg)
                                self.models[key] = keras.models.load_model(
                                    fpath, compile=False, custom_objects={'Dense': PatchedDense}
                                )
                                print(f"[OK] Loaded model {key} with PatchedDense")
                            except Exception as e2:
                                raise e2
                        self.model_types[key] = 'keras'
                    print(f"[OK] Loaded model {key}")
                    # record input size for Keras models
                    if self.model_types.get(key) == 'keras':
                        try:
                            ishape = self.models[key].input_shape
                            if isinstance(ishape, tuple) and len(ishape) == 4:
                                h, w = ishape[1], ishape[2]
                                if isinstance(h, int) and isinstance(w, int):
                                    self.input_sizes[key] = (h, w)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[WARNING] Could not load {fname}: {str(e)}")
            labels_path = os.path.join(models_dir, 'class_labels.json')
            if os.path.exists(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k in ['rice_potato', 'corn_blackgram', 'tomato_cotton', 'wheat_pumpkin']:
                        self.labels[k] = data.get(k, [])
                print("[OK] Labels loaded")
            else:
                print("[WARNING] class_labels.json not found")
        except Exception as e:
            print(f"[ERROR] Failed to initialize models: {str(e)}")
        try:
            base_dir = os.path.dirname(__file__)
            models_dir = os.path.join(base_dir, 'models')
            self.crop_classifier = None
            self.crop_input_size = (224, 224)
            crop_path = None
            for name in ['crop_classifier_new.h5', 'crop_classifier.h5']:
                p = os.path.join(models_dir, name)
                if os.path.exists(p):
                    crop_path = p
                    break
            if crop_path:
                self.crop_classifier = keras.models.load_model(crop_path, compile=False)
                try:
                    ishape = self.crop_classifier.input_shape
                    if isinstance(ishape, tuple) and len(ishape) == 4:
                        h, w = ishape[1], ishape[2]
                        if isinstance(h, int) and isinstance(w, int):
                            self.crop_input_size = (h, w)
                except Exception:
                    pass
                names_path = os.path.join(models_dir, 'class_names.json')
                self.crop_names = []
                if os.path.exists(names_path):
                    with open(names_path, 'r', encoding='utf-8') as f:
                        self.crop_names = json.load(f)
                print("[OK] Crop classifier loaded")
            else:
                print("[WARNING] Crop classifier not found")
        except Exception as e:
            print(f"[WARNING] Failed to initialize crop classifier: {str(e)}")
    
    def predict(self, image):
        try:
            if not self.models:
                return {
                    'success': False,
                    'error': 'Working model not available',
                    'disease': 'Unknown',
                    'confidence': 0.0,
                    'traditionalFormat': True
                }

            # Normalize incoming image to a base PIL Image
            if isinstance(image, np.ndarray):
                arr = image
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.clip(arr, 0, 255).astype('uint8')
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                base_img = Image.fromarray(arr)
            elif isinstance(image, Image.Image):
                base_img = image
            else:
                return {
                    'success': False,
                    'error': 'Unsupported image format',
                    'disease': 'Unknown',
                    'confidence': 0.0,
                    'traditionalFormat': True
                }
            if base_img.mode != 'RGB':
                base_img = base_img.convert('RGB')
            def _norm_key_for(mk):
                nk = mk.lower().replace('&', 'and')
                if '(' in mk and ')' in mk:
                    inside = mk[mk.find('(')+1:mk.find(')')].lower().replace(' and ', '_').replace(' ', '_')
                    nk = inside
                if 'rice' in nk and 'potato' in nk:
                    return 'rice_potato'
                if 'corn' in nk and 'blackgram' in nk:
                    return 'corn_blackgram'
                if 'tomato' in nk and 'cotton' in nk:
                    return 'tomato_cotton'
                if 'wheat' in nk and 'pumpkin' in nk:
                    return 'wheat_pumpkin'
                if 'model4' in nk or 'model_4' in nk:
                    return 'wheat_pumpkin'
                if 'model3' in nk or 'model_3' in nk:
                    return 'tomato_cotton'
                if 'model2' in nk or 'model_2' in nk:
                    return 'corn_blackgram'
                if 'model1' in nk or 'model_1' in nk:
                    return 'rice_potato'
                return nk
            best = {'model_key': None, 'class_idx': -1, 'confidence': -1.0, 'chosen_crop': None}
            for key, model in self.models.items():
                try:
                    # per-model resize
                    if self.model_types.get(key) == 'keras':
                        size = self.input_sizes.get(key, (224, 224))
                        img_k = base_img.resize(size)
                        arr_k = np.array(img_k, dtype='float32')
                        nk = _norm_key_for(key)
                        if nk == 'corn_blackgram':
                            try:
                                from keras.applications.efficientnet import preprocess_input as ef_pre
                            except Exception:
                                try:
                                    from tensorflow.keras.applications.efficientnet import preprocess_input as ef_pre
                                except Exception:
                                    ef_pre = None
                            if ef_pre is not None:
                                arr_k = ef_pre(arr_k)
                            else:
                                arr_k = arr_k / 255.0
                        elif nk == 'rice_potato':
                            try:
                                from keras.applications.resnet50 import preprocess_input as rn_pre
                            except Exception:
                                try:
                                    from tensorflow.keras.applications.resnet50 import preprocess_input as rn_pre
                                except Exception:
                                    rn_pre = None
                            if rn_pre is not None:
                                arr_k = rn_pre(arr_k)
                            else:
                                arr_k = arr_k / 255.0
                        else:
                            arr_k = arr_k / 255.0
                        arr_k = np.expand_dims(arr_k, axis=0)
                        preds = model.predict(arr_k, verbose=0)[0]
                    else:
                        nk = _norm_key_for(key)
                        size = (300, 300) if nk == 'tomato_cotton' else (224, 224)
                        img_t = base_img.resize(size)
                        arr_t = np.array(img_t, dtype='float32') / 255.0
                        arr_t = np.expand_dims(arr_t, axis=0)
                        x = torch.from_numpy(arr_t).permute(0, 3, 1, 2)
                        if nk == 'tomato_cotton':
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                            x = (x - mean) / std
                        with torch.no_grad():
                            out = model(x)
                            if isinstance(out, (list, tuple)):
                                out = out[0]
                            probs = F.softmax(out, dim=1)
                            preds = probs.cpu().numpy()[0]
                    nk = _norm_key_for(key)
                    labels_for_model = self.labels.get(nk, [])
                    crop_a, crop_b = None, None
                    if nk == 'rice_potato':
                        crop_a, crop_b = 'Rice', 'Potato'
                    elif nk == 'corn_blackgram':
                        crop_a, crop_b = 'Corn', 'Blackgram'
                    elif nk == 'tomato_cotton':
                        crop_a, crop_b = 'Tomato', 'Cotton'
                    elif nk == 'wheat_pumpkin':
                        crop_a, crop_b = 'Wheat', 'Pumpkin'
                    if labels_for_model and crop_a and crop_b:
                        idx_a = [i for i, l in enumerate(labels_for_model) if l.startswith(crop_a + "_")]
                        idx_b = [i for i, l in enumerate(labels_for_model) if l.startswith(crop_b + "_")]
                        sum_a = float(np.sum(preds[idx_a])) if idx_a else 0.0
                        sum_b = float(np.sum(preds[idx_b])) if idx_b else 0.0
                        if sum_a >= sum_b and idx_a:
                            local_idx = idx_a[int(np.argmax(preds[idx_a]))]
                            idx = int(local_idx)
                            conf = float(preds[idx])
                            chosen_crop = crop_a
                        elif idx_b:
                            local_idx = idx_b[int(np.argmax(preds[idx_b]))]
                            idx = int(local_idx)
                            conf = float(preds[idx])
                            chosen_crop = crop_b
                        else:
                            idx = int(np.argmax(preds))
                            conf = float(preds[idx])
                            chosen_crop = None
                    else:
                        idx = int(np.argmax(preds))
                        conf = float(preds[idx])
                        chosen_crop = None
                    if conf > best['confidence']:
                        best = {'model_key': key, 'class_idx': idx, 'confidence': conf, 'chosen_crop': chosen_crop}
                except Exception as e:
                    print(f"[WARNING] Prediction failed for {key}: {str(e)}")
                    continue
            if best['model_key'] is None:
                return {
                    'success': False,
                    'error': 'All models failed to predict',
                    'disease': 'Unknown',
                    'confidence': 0.0,
                    'traditionalFormat': True
                }
            class_idx = best['class_idx']
            confidence = best['confidence']
            if confidence < 0.6:
                return {
                    'success': True,
                    'disease': 'Healthy',
                    'confidence': confidence,
                    'cropName': 'Unknown',
                    'severity': 'Low',
                    'description': f'No clear disease detected. Confidence {confidence*100:.1f}%.',
                    'traditionalFormat': True
                }
            mk = best['model_key']
            norm_key = mk.lower()
            if '(' in mk and ')' in mk:
                inside = mk[mk.find('(')+1:mk.find(')')].lower().replace(' and ', '_').replace(' ', '_')
                norm_key = inside
            norm_key = norm_key.replace('&', 'and')
            if 'rice' in norm_key and 'potato' in norm_key:
                norm_key = 'rice_potato'
            elif 'corn' in norm_key and 'blackgram' in norm_key:
                norm_key = 'corn_blackgram'
            elif 'tomato' in norm_key and 'cotton' in norm_key:
                norm_key = 'tomato_cotton'
            elif 'wheat' in norm_key and 'pumpkin' in norm_key:
                norm_key = 'wheat_pumpkin'
            elif 'model4' in norm_key or 'model_4' in norm_key:
                norm_key = 'wheat_pumpkin'
            elif 'model3' in norm_key or 'model_3' in norm_key:
                norm_key = 'tomato_cotton'
            elif 'model2' in norm_key or 'model_2' in norm_key:
                norm_key = 'corn_blackgram'
            elif 'model1' in norm_key or 'model_1' in norm_key:
                norm_key = 'rice_potato'
            labels_for_model = self.labels.get(norm_key, [])
            if labels_for_model and class_idx < len(labels_for_model):
                disease_name = labels_for_model[class_idx]
            else:
                disease_name = f"Disease_{class_idx}"
            crop_name = 'Unknown'
            if 'rice_potato' in norm_key:
                crop_name = best.get('chosen_crop') or ('Rice' if 'Rice_' in disease_name else ('Potato' if 'Potato_' in disease_name else 'Rice'))
            elif 'corn_blackgram' in norm_key:
                crop_name = best.get('chosen_crop') or ('Corn' if 'Corn_' in disease_name else ('Blackgram' if 'Blackgram_' in disease_name else 'Corn'))
            elif 'tomato_cotton' in norm_key:
                crop_name = best.get('chosen_crop') or ('Tomato' if 'Tomato_' in disease_name else ('Cotton' if 'Cotton_' in disease_name else 'Tomato'))
            elif 'wheat_pumpkin' in norm_key:
                crop_name = best.get('chosen_crop') or ('Wheat' if 'Wheat_' in disease_name else ('Pumpkin' if 'Pumpkin_' in disease_name else 'Wheat'))
            if 'Potato_' in disease_name:
                crop_name = 'Potato'
            elif 'Rice_' in disease_name:
                crop_name = 'Rice'
            elif disease_name == 'Healthy':
                crop_name = 'Unknown'
            return {
                'success': True,
                'disease': disease_name,
                'confidence': confidence,
                'cropName': crop_name,
                'severity': 'Low' if disease_name == 'Healthy' else ('High' if confidence > 0.8 else 'Medium'),
                'description': f'{crop_name} - {disease_name} detected with {confidence*100:.1f}% confidence.',
                'traditionalFormat': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Working prediction failed: {str(e)}',
                'disease': 'Unknown',
                'confidence': 0.0,
                'traditionalFormat': True
            }
    def classify_crop(self, image):
        try:
            if isinstance(image, np.ndarray):
                arr = image
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.clip(arr, 0, 255).astype('uint8')
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                base_img = Image.fromarray(arr)
            elif isinstance(image, Image.Image):
                base_img = image
            else:
                return {'success': False, 'error': 'Unsupported image format'}
            if base_img.mode != 'RGB':
                base_img = base_img.convert('RGB')
            if not self.crop_classifier:
                return {'success': False, 'error': 'Crop classifier not available'}
            img_c = base_img.resize(self.crop_input_size)
            arr_c = np.array(img_c, dtype='float32')
            ef_pre = None
            try:
                from keras.applications.efficientnet import preprocess_input as ef_pre
            except Exception:
                try:
                    from tensorflow.keras.applications.efficientnet import preprocess_input as ef_pre
                except Exception:
                    ef_pre = None
            if ef_pre is not None:
                arr_c = ef_pre(arr_c)
            else:
                arr_c = arr_c / 255.0
            arr_c = np.expand_dims(arr_c, axis=0)
            preds = self.crop_classifier.predict(arr_c, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            crop = self.crop_names[idx] if self.crop_names and idx < len(self.crop_names) else f'Crop_{idx}'
            return {'success': True, 'crop': crop, 'confidence': conf, 'index': idx}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    def predict_with_model(self, image, target_norm_key, selected_crop=None):
        try:
            if isinstance(image, np.ndarray):
                arr = image
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.clip(arr, 0, 255).astype('uint8')
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                base_img = Image.fromarray(arr)
            elif isinstance(image, Image.Image):
                base_img = image
            else:
                return {'success': False, 'error': 'Unsupported image format'}
            if base_img.mode != 'RGB':
                base_img = base_img.convert('RGB')
            found = None
            for key in self.models.keys():
                nk = key.lower()
                if '(' in key and ')' in key:
                    inside = key[key.find('(')+1:key.find(')')].lower().replace(' and ', '_').replace(' ', '_')
                    nk = inside
                nk = nk.replace('&', 'and').replace('.h5', '').replace('.keras', '').replace('.pth', '')
                if 'rice' in nk and 'potato' in nk:
                    nk = 'rice_potato'
                elif 'corn' in nk and 'blackgram' in nk:
                    nk = 'corn_blackgram'
                elif 'tomato' in nk and 'cotton' in nk:
                    nk = 'tomato_cotton'
                elif 'wheat' in nk and 'pumpkin' in nk:
                    nk = 'wheat_pumpkin'
                if nk == target_norm_key:
                    found = key
                    break
            if not found:
                return {'success': False, 'error': f'Model for {target_norm_key} not found'}
            key = found
            model = self.models[key]
            nk = target_norm_key
            if self.model_types.get(key) == 'keras':
                size = self.input_sizes.get(key, (224, 224))
                img_k = base_img.resize(size)
                arr_k = np.array(img_k, dtype='float32')
                if nk == 'corn_blackgram':
                    try:
                        from keras.applications.efficientnet import preprocess_input as ef_pre
                    except Exception:
                        try:
                            from tensorflow.keras.applications.efficientnet import preprocess_input as ef_pre
                        except Exception:
                            ef_pre = None
                    if ef_pre is not None:
                        arr_k = ef_pre(arr_k)
                    else:
                        arr_k = arr_k / 255.0
                elif nk == 'rice_potato':
                    try:
                        from keras.applications.resnet50 import preprocess_input as rn_pre
                    except Exception:
                        try:
                            from tensorflow.keras.applications.resnet50 import preprocess_input as rn_pre
                        except Exception:
                            rn_pre = None
                    if rn_pre is not None:
                        arr_k = rn_pre(arr_k)
                    else:
                        arr_k = arr_k / 255.0
                elif nk == 'wheat_pumpkin':
                    try:
                        from keras.applications.efficientnet import preprocess_input as ef_pre
                    except Exception:
                        try:
                            from tensorflow.keras.applications.efficientnet import preprocess_input as ef_pre
                        except Exception:
                            ef_pre = None
                    if ef_pre is not None:
                        arr_k = ef_pre(arr_k)
                    else:
                        arr_k = arr_k / 255.0
                else:
                    arr_k = arr_k / 255.0
                arr_k = np.expand_dims(arr_k, axis=0)
                preds = model.predict(arr_k, verbose=0)[0]
            else:
                size = (300, 300) if nk == 'tomato_cotton' else (224, 224)
                img_t = base_img.resize(size)
                arr_t = np.array(img_t, dtype='float32') / 255.0
                arr_t = np.expand_dims(arr_t, axis=0)
                x = torch.from_numpy(arr_t).permute(0, 3, 1, 2)
                if nk == 'tomato_cotton':
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    x = (x - mean) / std
                with torch.no_grad():
                    out = model(x)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    probs = F.softmax(out, dim=1)
                    preds = probs.cpu().numpy()[0]
            labels_for_model = self.labels.get(nk, [])
            if selected_crop:
                sc = selected_crop.lower()
                if nk == 'rice_potato':
                    target_prefix = 'Rice_' if 'rice' in sc else 'Potato_'
                elif nk == 'corn_blackgram':
                    target_prefix = 'Corn_' if 'corn' in sc else 'Blackgram_'
                elif nk == 'tomato_cotton':
                    target_prefix = 'Tomato_' if 'tomato' in sc else 'Cotton_'
                elif nk == 'wheat_pumpkin':
                    target_prefix = 'Wheat_' if 'wheat' in sc else 'Pumpkin_'
                else:
                    target_prefix = None
                if target_prefix:
                    idxs = [i for i, l in enumerate(labels_for_model) if l.startswith(target_prefix) or l == 'Healthy' or l.endswith('_Healthy')]
                    if idxs:
                        local = preds[idxs]
                        li = int(np.argmax(local))
                        idx = int(idxs[li])
                        conf = float(preds[idx])
                        # build top-3 candidates within subset
                        order = np.argsort(local)[::-1]
                        top = []
                        for oi in order[:3]:
                            gi = int(idxs[int(oi)])
                            name = labels_for_model[gi] if labels_for_model and gi < len(labels_for_model) else f"Disease_{gi}"
                            top.append({'disease': name, 'confidence': float(preds[gi])})
                        top_candidates = top
                    else:
                        idx = int(np.argmax(preds))
                        conf = float(preds[idx])
                        top_candidates = []
                else:
                    idx = int(np.argmax(preds))
                    conf = float(preds[idx])
                    top_candidates = []
            else:
                idx = int(np.argmax(preds))
                conf = float(preds[idx])
                top_candidates = []
            disease_name = labels_for_model[idx] if labels_for_model and idx < len(labels_for_model) else f"Disease_{idx}"
            crop_name = selected_crop or 'Unknown'
            return {
                'success': True,
                'disease': disease_name,
                'confidence': conf,
                'cropName': crop_name,
                'severity': 'Low' if disease_name == 'Healthy' else ('High' if conf > 0.8 else 'Medium'),
                'description': f'{crop_name} - {disease_name} detected with {conf*100:.1f}% confidence.',
                'traditionalFormat': True,
                'topCandidates': top_candidates
            }
        except Exception as e:
            return {'success': False, 'error': f'Predict with model failed: {str(e)}'}
    def predict_two_stage(self, image):
        first = self.classify_crop(image)
        if not first.get('success'):
            return {'success': False, 'error': first.get('error', 'Crop classification failed'), 'stage': 'crop'}
        crop = first['crop']
        crop_lower = crop.lower()
        if 'rice' in crop_lower or 'potato' in crop_lower:
            target = 'rice_potato'
        elif 'corn' in crop_lower or 'blackgram' in crop_lower:
            target = 'corn_blackgram'
        elif 'tomato' in crop_lower or 'cotton' in crop_lower:
            target = 'tomato_cotton'
        elif 'wheat' in crop_lower or 'pumpkin' in crop_lower:
            target = 'wheat_pumpkin'
        else:
            return {'success': False, 'error': f'Unknown crop: {crop}', 'stage': 'crop'}
        second = self.predict_with_model(image, target, crop)
        if not second.get('success'):
            return {'success': False, 'error': second.get('error', 'Disease prediction failed'), 'stage': 'disease'}
        out = dict(second)
        out['cropName'] = crop
        return out

# Global instance
working_model_manager = None

def get_working_model_manager():
    """Get working model manager"""
    global working_model_manager
    if working_model_manager is None:
        working_model_manager = WorkingModelManager()
        print("[OK] Working model manager initialized")
    return working_model_manager
