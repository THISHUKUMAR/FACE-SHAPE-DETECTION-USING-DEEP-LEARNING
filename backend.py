# # backend.py
# import torch
# from torch import nn
# from torchvision import transforms
# from PIL import Image
# import pickle
# import matplotlib.pyplot as plt

# class TinyVGG(nn.Module):
#     def __init__(self, input_channels: int, hidden_units: int, output_shape: int, dropout_p: float = 0.3):
#         """
#         Tiny VGG-like CNN for image classification.

#         Args:
#             input_channels: Number of channels in input images (3 for RGB)
#             hidden_units: Number of channels in conv layers
#             output_shape: Number of classes
#             dropout_p: Dropout probability before classifier
#         """
#         super().__init__()

#         # --- Block 1 ---
#         self.block1 = nn.Sequential(
#             nn.Conv2d(input_channels, hidden_units, kernel_size=3, padding=1),
#             nn.BatchNorm2d(hidden_units),
#             nn.ReLU(),
#             nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
#             nn.BatchNorm2d(hidden_units),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )

#         # --- Block 2 ---
#         self.block2 = nn.Sequential(
#             nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
#             nn.BatchNorm2d(hidden_units),
#             nn.ReLU(),
#             nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
#             nn.BatchNorm2d(hidden_units),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )

#         # --- Adaptive pooling to fix feature map size ---
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

#         # --- Classifier ---
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(dropout_p),
#             nn.Linear(hidden_units * 7 * 7, output_shape)
#         )

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.adaptive_pool(x)
#         x = self.classifier(x)
#         return x


# device = "cuda" if torch.cuda.is_available() else "cpu"


# # ---------------- Load class names
# with open("class_names.pkl", "rb") as f:
#     class_names = pickle.load(f)

# # ---------------- Create model and load weights
# model = TinyVGG(input_channels=3, hidden_units=64, output_shape=len(class_names))
# model.load_state_dict(torch.load("face_shape_model.pth", map_location=device))
# model.to(device)
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ])

# # ---------------- Prediction function
# # def predict_image(image_path):
# #     img = Image.open(image_path).convert("RGB")
# #     img_tensor = transform(img).unsqueeze(0).to(device)

# #     with torch.no_grad():
# #         logits = model(img_tensor)
# #         probs = torch.softmax(logits, dim=1)
# #         pred_label = torch.argmax(probs, dim=1).item()
# #         pred_prob = probs.max().item()

# #     print(f"Prediction: {class_names[pred_label]} | Confidence: {pred_prob:.3f}")

# #     # Optional: display the image
# #     plt.imshow(img)
# #     plt.title(f"Pred: {class_names[pred_label]} | Prob: {pred_prob:.3f}")
# #     plt.axis("off")
# #     plt.show()


# def predict_image(image_path: str):
#     """Predicts the class of an image and plots it. Only input required is image path."""
    
#     # Load image
#     img = Image.open(image_path).convert("RGB")
    
#     # Apply transform
#     img_tensor = transform(img).unsqueeze(0).to(device)
    
#     # Inference
#     model.eval()
#     with torch.no_grad():
#         logits = model(img_tensor)
#         probs = torch.softmax(logits, dim=1)
#         pred_label = torch.argmax(probs, dim=1).item()
#         pred_prob = probs.max().item()
    
#     # Plot
#     plt.imshow(img)
#     plt.title(f"Pred: {class_names[pred_label]} | Prob: {pred_prob:.3f}")
#     plt.axis("off")
#     plt.show()

# backend.py
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# ---------------- TinyVGG model definition (same as notebook) ----------------
class TinyVGG(nn.Module):
    def __init__(self, input_channels: int, hidden_units: int, output_shape: int, dropout_p: float = 0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_units * 7 * 7, output_shape)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Load class names ----------------
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# ---------------- Create model and load trained weights ----------------
model = TinyVGG(input_channels=3, hidden_units=64, output_shape=len(class_names))
model.load_state_dict(torch.load("face_shape_model.pth", map_location=device))
model.to(device)
model.eval()  # IMPORTANT: evaluation mode

# ---------------- Define deterministic transform (same as notebook) ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- Prediction function ----------------
def predict_image(image_path: str):
    """Predict class of image with exact notebook transforms"""
    # Load image
    img = Image.open(image_path).convert("RGB")
    # Apply transform
    img_tensor = transform(img).unsqueeze(0).to(device)
    # Model inference
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        pred_prob = probs.max().item()
    # Plot image
    plt.imshow(img)
    plt.title(f"Pred: {class_names[pred_label]} | Prob: {pred_prob:.3f}")
    plt.axis("off")
    plt.show()
