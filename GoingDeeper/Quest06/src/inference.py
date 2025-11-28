import torch
import numpy as np
from PIL import Image
import cv2
from tqdm.auto import tqdm
import easyocr
import matplotlib.pyplot as plt
from src.utils import decode_greedy, compute_metric
import torch.nn.functional as F


def check_inference(model, dataset, label_converter, device, num_samples=5):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        # MJDataset returns: padded_img, encoded_label, len(encoded_label), label
        image, label_encoded, length, label_str = dataset[idx]
        
        # image is numpy array (C, H, W), normalized 0-1
        # Need to convert to tensor first
        img_tensor = torch.from_numpy(image).unsqueeze(0).to(device) # (1, C, H, W)
        
        with torch.no_grad():
            output = model(img_tensor) # (T, B, C)
            pred_str = decode_greedy(output, label_converter)[0]
        
        # Ground truth label
        gt_str = label_str # Use the raw label string returned by dataset
        
        # Visualization
        # Denormalize image (0-1 -> 0-255)
        # image is numpy array (C, H, W)
        img_np = image.transpose(1, 2, 0)
        img_np = img_np * 255.0
        img_np = img_np.clip(0, 255).astype(np.uint8)
        
        plt.figure(figsize=(5, 2))
        plt.imshow(img_np)
        plt.title(f"GT: {gt_str} | Pred: {pred_str}")
        plt.axis('off')
        plt.show()

def test_model(model, test_loader, label_converter, device):
    model.eval()
    
    total_correct = 0
    total_distance = 0
    total_samples = 0
    total_loss = 0.0
    
    criterion = torch.nn.CTCLoss(blank=0, reduction='sum')  # sum으로 나중에 평균 내기 쉽게
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, labels_encoded, input_lengths, label_lengths, label_strings = batch
            
            images = images.to(device)
            labels_encoded = labels_encoded.to(device)
            label_lengths = label_lengths.to(device)
            
            outputs = model(images)  # (T, B, C)
            log_probs = F.log_softmax(outputs, dim=2)
            
            # input_lengths 계산 (대부분의 CRNN은 T 그대로 나옴)
            batch_size = images.size(0)
            input_lengths_tensor = torch.full((batch_size,), log_probs.size(0), 
                                            dtype=torch.long, device=device)
            
            # Loss 계산
            loss = criterion(log_probs, labels_encoded, input_lengths_tensor, label_lengths)
            total_loss += loss.item()
            
            # 기존 metric 계산
            correct, distance, batch_size = compute_metric(
                outputs, labels_encoded, label_lengths, label_converter
            )
            
            total_correct += correct
            total_distance += distance
            total_samples += batch_size
    
    # 최종 결과
    accuracy = total_correct / total_samples * 100.0
    avg_edit_distance = total_distance / total_samples
    avg_loss = total_loss / total_samples  # sample당 평균 CTC loss
    
    print(f"\n=== Test Result ===")
    print(f"Total samples   : {total_samples}")
    print(f"Accuracy        : {accuracy:.2f}% ({total_correct}/{total_samples})")
    print(f"Avg Edit Dist   : {avg_edit_distance:.3f}")
    print(f"Avg CTC Loss    : {avg_loss:.4f}")
    
    return {
        'accuracy': accuracy,
        'avg_edit_distance': avg_edit_distance,
        'avg_loss': avg_loss,
        'total_samples': total_samples
    }

def detect_text(img_path):
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(img_path)
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    cropped_imgs = []
    for (bbox, text, prob) in result:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        
        cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        
        # Crop
        crop = img[tl[1]:br[1], tl[0]:br[0]]
        if crop.size > 0:
            cropped_imgs.append(crop)
            
    return img, cropped_imgs, result

def recognize_img(model, cropped_imgs, label_converter, device, imgH=32, imgW=100):
    model.eval()
    results = []
    
    for i, crop in enumerate(cropped_imgs):
        h, w, c = crop.shape
        if h == 0 or w == 0:
            continue
        
        # Preprocessing
        img_pil = Image.fromarray(crop)
        
        orig_w, orig_h = img_pil.size
        target_width = min(int(orig_w * imgH / orig_h), imgW)
        target_img_size = (target_width, imgH)
        img_pil = img_pil.resize(target_img_size)
        
        img_np = np.array(img_pil)
        img = img_np.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        
        padded_img = np.zeros((3, imgH, imgW), dtype=np.float32)
        c, h, w_new = img.shape
        padded_img[:, :h, :w_new] = img
        
        img_tensor = torch.tensor(padded_img[np.newaxis, ...], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred_str = decode_greedy(output, label_converter)[0]
            results.append(pred_str)
            
        plt.figure(figsize=(3, 1))
        plt.imshow(crop)
        plt.title(f"Pred: {pred_str}")
        plt.axis('off')
        plt.show()
        
    return results
