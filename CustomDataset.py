import torch
from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def parse_annotation(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        annotations = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            annotations.append((name, xmin, ymin, xmax, ymax))
        
        return annotations
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return []

def load_image(image_path):
    try:
        image = cv2.imread(image_path)
        return image
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

class VehicleDataset(Dataset):
    def __init__(self, images_folder, annotations_folder, transform=None):
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.transform = transform
        
        # Create list of image file paths
        self.images_files = [os.path.join(root, name)
                             for root, dirs, files in os.walk(images_folder)
                             for name in files if name.endswith(".jpg")]
        
    def __len__(self):
        return len(self.images_files)
    
    def __getitem__(self, index):
        img_path = self.images_files[index]
        annotation_path = img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
        
        print(f"Original image path: {img_path}")
        print(f"Corresponding annotation path: {annotation_path}")
        
        # Load image and annotations
        image = load_image(img_path)
        annotations = parse_annotation(annotation_path)
        
        if image is None or not annotations:
            # Handle case where image or annotations are not loaded properly
            # Return None or handle as appropriate for your application
            return None, None
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, annotations

# Example usage
dataset = VehicleDataset(r'idd-detection\IDD_Detection\JPEGImages\frontFar', r'idd-detection\IDD_Detection\Annotations\frontFar')

# Accessing the first data sample
image, annotations = dataset[0]

# Check if image and annotations are loaded correctly
if image is not None and annotations is not None:
    print(f"Image shape: {image.shape}")
    print(f"Annotations: {annotations}")
else:
    print("Error loading data sample.")

# Optionally, add visualization function to display image with annotation

def display_image_with_annotations(image, annotations):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')

    for annotation in annotations:
        name, xmin, ymin, xmax, ymax = annotation
        bbox = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(bbox)
        plt.text(xmin, ymin - 5, name, fontsize=8, color='r')

    plt.show()

# Example visualization
if image is not None and annotations is not None:
    display_image_with_annotations(image, annotations)
