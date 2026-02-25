import os
import json
import torch
import timm

def convert_pytorch_to_torchscript(pth_path, output_ts_path, num_classes: int = None):
    print(f"[INFO] Loading PyTorch checkpoint: {pth_path}")
    if not os.path.exists(pth_path):
        print(f"[ERROR] .pth file not found: {pth_path}")
        return False
    try:
        ckpt = torch.load(pth_path, map_location='cpu')
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return False
    if num_classes is None:
        # Try to read from class_labels.json
        try:
            labels_path = os.path.join(os.path.dirname(pth_path), 'class_labels.json')
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'tomato_cotton' in data:
                num_classes = len(data['tomato_cotton'])
                print(f"[OK] num_classes inferred from class_labels.json: {num_classes}")
        except Exception:
            pass
    if num_classes is None:
        print("[ERROR] Could not infer num_classes. Please provide labels or num_classes.")
        return False
    print("[INFO] Creating timm model: tf_efficientnet_b3_ns")
    model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False, num_classes=num_classes)
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded state_dict. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        try:
            ckpt.eval()
            model = ckpt
            print("[OK] Using full model object from checkpoint")
        except Exception as e:
            print(f"[ERROR] Checkpoint is not a model and no state_dict available: {e}")
            return False
    model.eval()
    dummy = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        try:
            traced = torch.jit.trace(model, dummy)
        except Exception as e:
            print(f"[ERROR] TorchScript trace failed: {e}")
            return False
    torch.jit.save(traced, output_ts_path)
    print(f"[OK] TorchScript model saved to: {output_ts_path}")
    return True

if __name__ == '__main__':
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    pth_path = os.path.join(models_dir, 'Model3(Tomato and Cotton).pth')
    output_ts_path = os.path.join(models_dir, 'Model3(Tomato and Cotton).pt')
    convert_pytorch_to_torchscript(pth_path, output_ts_path)
