# --- PyTorch core ---
import torch                                     # Main PyTorch library
import torch.nn as nn                            # Needed to define the model architecture
from torch.utils.data import DataLoader          # Batches the test dataset for inference

# --- Torchvision ---
from torchvision import datasets                 # Provides OxfordIIITPet test split
from torchvision import transforms               # Image preprocessing pipeline

# ---------------------------------------------------------------------------
# Device
# Automatically use CUDA if available, otherwise CPU.
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---------------------------------------------------------------------------
# Config
# IMG_SIZE and NUM_CLASSES must match the values used in train.py exactly,
# since the saved model was built and trained with those dimensions.
# ---------------------------------------------------------------------------
IMG_SIZE    = 64
BATCH_SIZE  = 32
NUM_CLASSES = 37
DATA_ROOT   = './data'
MODEL_PATH  = 'model.pth'

# ---------------------------------------------------------------------------
# Transforms
# Identical to val_transforms in train.py — deterministic, no augmentation.
# ---------------------------------------------------------------------------
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Dataset
# Load the official test split — this is held out and never touched in train.py.
# ---------------------------------------------------------------------------
test_dataset = datasets.OxfordIIITPet(
    root=DATA_ROOT,
    split='test',
    transform=test_transforms,
    download=True           # Downloads the dataset if not already present
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # No shuffling — order doesn't matter for evaluation
    num_workers=0
)

print(f'Test images: {len(test_dataset)}')
print(f'Test batches: {len(test_loader)}')