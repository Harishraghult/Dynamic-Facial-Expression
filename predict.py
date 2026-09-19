"""
predict.py - Run inference on a single video clip using the trained M3DFEL model.

Usage:
    # Test on a random clip (auto-detected):
    python predict.py

    # Test on a specific clip folder (containing .jpg frames):
    python predict.py --clip_path "e:\Dynamic Facial Expression\Clip\clip_224x224\00001"

    # Resume from a specific checkpoint:
    python predict.py --resume outputs/DFEW-[09-14]-[20-45]/model_best.pth
"""

import torch
import glob
import os
import sys
import argparse
import numpy as np
import PIL.Image as Image
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model
from datasets.video_transform import GroupResize, Stack, ToTorchFormatTensor


EMOTIONS = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"]
EMOJI    = ["Happy", "Sad",   "Neutral", "Angry",  "Surprise", "Disgust", "Fear"]


def load_model(checkpoint_path, gpu_id=0):
    class Args:
        gpu_ids = [gpu_id] if torch.cuda.is_available() else []
        num_frames = 16
        instance_length = 4
        crop_size = 112
        num_classes = 7
        model = "r3d"

    args = Args()
    device = torch.device(f"cuda:{gpu_id}" if args.gpu_ids and torch.cuda.is_available() else "cpu")

    model = create_model(args)
    model.to(device)

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint["state_dict"]
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state)
        print(f"  => Epoch {checkpoint['epoch']} | Best WA: {checkpoint['best_wa']:.2%} | Best UA: {checkpoint['best_ua']:.2%}")
    else:
        print("WARNING: No checkpoint. Using random weights.")

    model.eval()
    return model, device


def load_clip(clip_folder, num_frames=16, image_size=112):
    frame_paths = sorted(glob.glob(os.path.join(clip_folder, "*.jpg")))
    if len(frame_paths) == 0:
        raise ValueError(f"No .jpg frames found in: {clip_folder}")

    total = len(frame_paths)
    indices = [int(total * i / num_frames) for i in range(num_frames)]
    selected = [frame_paths[min(i, total - 1)] for i in indices]

    transform = torchvision.transforms.Compose([
        GroupResize(image_size),
        Stack(),
        ToTorchFormatTensor()
    ])

    images = [Image.open(p).convert("RGB") for p in selected]
    tensor = transform(images)
    tensor = torch.reshape(tensor, (-1, 3, image_size, image_size))
    return tensor.unsqueeze(0)


def predict_clip(model, device, clip_folder):
    tensor = load_clip(clip_folder).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def find_best_checkpoint():
    checkpoints = glob.glob("outputs/**/model_best.pth", recursive=True)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="M3DFEL Prediction")
    parser.add_argument("--clip_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="e:\\Dynamic Facial Expression")
    args = parser.parse_args()

    checkpoint_path = args.resume or find_best_checkpoint()
    if not checkpoint_path:
        print("ERROR: No checkpoint found. Train first or pass --resume <path>")
        return

    model, device = load_model(checkpoint_path, args.gpu_id)

    if not args.clip_path:
        clip_base = os.path.join(args.data_root, "Clip", "clip_224x224")
        all_clips = sorted([d for d in os.listdir(clip_base) if os.path.isdir(os.path.join(clip_base, d))])
        if not all_clips:
            print(f"ERROR: No clips found at {clip_base}. Pass --clip_path.")
            return
        import random
        chosen = random.choice(all_clips)
        clip_folder = os.path.join(clip_base, chosen)
        print(f"\nNo clip specified. Picking random: {chosen}")
    else:
        clip_folder = args.clip_path

    print(f"Clip  : {clip_folder}")
    print(f"Frames: {len(glob.glob(os.path.join(clip_folder, '*.jpg')))}")

    pred_idx, probs = predict_clip(model, device, clip_folder)

    print("\n" + "=" * 45)
    print(f"  Prediction --> {EMOTIONS[pred_idx].upper()}")
    print("=" * 45)
    print("\nClass probabilities:")
    for i, (emo, prob) in enumerate(zip(EMOTIONS, probs)):
        bar = "#" * int(prob * 30)
        marker = " <-- predicted" if i == pred_idx else ""
        print(f"  {emo:<10} {prob:.4f}  {bar}{marker}")


if __name__ == "__main__":
    main()
