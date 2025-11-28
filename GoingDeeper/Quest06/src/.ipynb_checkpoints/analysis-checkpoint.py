import lmdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import six
from tqdm import tqdm
from IPython.display import display

def analyze_text_length_distribution(dataset_path, max_text_len=None):
    """
    Analyzes and plots the distribution of text label lengths in the dataset.
    """
    env = lmdb.open(dataset_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    
    label_lengths = []
    
    with env.begin(write=False) as txn:
        num_samples = int(txn.get("num-samples".encode()))
        print(f"Analyzing {num_samples} samples for text length...")
        
        for i in tqdm(range(num_samples)):
            index = i + 1
            label_key = f"label-{index:09d}".encode()
            label = txn.get(label_key).decode("utf-8")
            label_lengths.append(len(label))
            
    env.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(label_lengths, bins=range(min(label_lengths), max(label_lengths) + 2), edgecolor='black', alpha=0.7)
    plt.title('Text Label Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Count')
    if max_text_len:
        plt.axvline(x=max_text_len, color='r', linestyle='--', label=f'MAX_TEXT_LEN={max_text_len}')
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Max Length: {max(label_lengths)}")
    print(f"Average Length: {np.mean(label_lengths):.2f}")

def analyze_image_distribution(dataset_path):
    """
    Analyzes and plots the distribution of image widths, heights, and aspect ratios.
    """
    env = lmdb.open(dataset_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    
    widths = []
    heights = []
    aspect_ratios = []
    
    with env.begin(write=False) as txn:
        num_samples = int(txn.get("num-samples".encode()))
        print(f"Analyzing {num_samples} samples for image statistics...")
        
        for i in tqdm(range(num_samples)):
            index = i + 1
            img_key = f"image-{index:09d}".encode()
            imgbuf = txn.get(img_key)
            
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            
            try:
                img = Image.open(buf)
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
            except IOError:
                continue
                
    env.close()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(widths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Image Width Distribution')
    plt.xlabel('Width')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 2)
    plt.hist(heights, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    plt.title('Image Height Distribution')
    plt.xlabel('Height')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 3)
    plt.hist(aspect_ratios, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.title('Aspect Ratio (W/H) Distribution')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Max Width: {max(widths)}, Min Width: {min(widths)}, Avg Width: {np.mean(widths):.2f}")
    print(f"Max Height: {max(heights)}, Min Height: {min(heights)}, Avg Height: {np.mean(heights):.2f}")
    print(f"Max AR: {max(aspect_ratios):.2f}, Min AR: {min(aspect_ratios):.2f}, Avg AR: {np.mean(aspect_ratios):.2f}")

def visualize_dataset_samples(dataset_path, num_samples=4):
    """
    Visualizes random samples from the dataset.
    """
    env = lmdb.open(dataset_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    
    with env.begin(write=False) as txn:
        for index in range(1, num_samples + 1):
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                img = Image.new('RGB', (100, 32))
                label = '-'

            width, height = img.size
            print('original image width:{}, height:{}'.format(width, height))

            target_width = min(int(width*32/height), 100)
            target_img_size = (target_width,32)
            print('target_img_size:{}'.format(target_img_size))
            img = np.array(img.resize(target_img_size)).transpose(1,0,2)

            print('display img shape:{}'.format(img.shape))
            print('label:{}'.format(label))
            display(Image.fromarray(img.transpose(1,0,2).astype(np.uint8)))
            
    env.close()

def visualize_augmented_samples(data_loader, num_samples=8):
    """
    Visualizes augmented samples from a DataLoader.
    """
    data_iter = iter(data_loader)
    images, encoded_labels, input_lengths, label_lengths, raw_labels = next(data_iter)

    vis_images_tensor = images[:num_samples]
    vis_labels = raw_labels[:num_samples]

    # 이미지 역정규화 및 차원 변환 (Tensor(C,H,W) -> Numpy(H,W,C))
    # Dataset에서 0~1로 정규화했으므로 다시 255를 곱해줍니다.
    vis_images_np = vis_images_tensor.numpy() * 255.0
    vis_images_np = vis_images_np.clip(0, 255).astype(np.uint8)
    vis_images_np = vis_images_np.transpose(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

    print(f"Augmented Images Samples (Weak Augmentation Applied):")
    
    # Calculate grid size
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    plt.figure(figsize=(20, 3 * rows))
    for i in range(num_samples):
        plt.subplot(rows, cols, i+1)
        plt.imshow(vis_images_np[i])
        plt.title(f"Label: {vis_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
