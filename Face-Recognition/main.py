import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, Normalize


# Model Definition
class FaceRecognition(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 224->112
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112->56
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56->28
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28->14
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = FaceRecognition()
model.load_state_dict(torch.load('face_recognition_model.pth', map_location=torch.device('cpu')))
print(sum(p.numel() for p in model.parameters()))
IMAGE_SIZE = 224
mean = (0.5227308869361877, 0.46569907665252686, 0.4313724637031555)
stddev = (0.3161585273687047, 0.3008185772213572, 0.2996168262495128)
transform = Compose([
    Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ToTensor(),
    Normalize(mean, stddev),
])

model.eval()
with torch.no_grad():
    for i in range(435):
        image = Image.open('faces_positive/frame_{i:03d}.jpg').convert('RGB')
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        outputs = model(image_tensor)
        predicted = (outputs.data > 0).int()
        if predicted != 1:
            print("FAILURE") # Never fails - overfit!!