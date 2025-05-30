import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

state_dict_path = "final_trained_model_state_dict.pth"
image_path = "directory of image"

class Model(nn.Module):
    def __init__(self, base_arch='resnet50'):
        super(Model, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', base_arch, pretrained=False)
        self.backbone.fc = nn.Identity()
        self.temp_head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 5))
        self.time_head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 5))

    def forward(self, x):
        features = self.backbone(x)
        temp_out = self.temp_head(features)
        time_out = self.time_head(features)
        return temp_out, time_out


Temp_mapping = {0: 650, 1: 700, 2: 750, 3: 800, 4: 850}
Time_mapping = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict



model = Model().to(device)
state_dict = torch.load(state_dict_path, map_location=device)
state_dict = fix_state_dict(state_dict)
model.load_state_dict(state_dict)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        temp_out, time_out = model(image)
        temp_pred = temp_out.argmax(dim=1).item()
        time_pred = time_out.argmax(dim=1).item()

    predicted_temp = Temp_mapping[temp_pred]
    predicted_time = Time_mapping[time_pred]
    return predicted_temp, predicted_time


if __name__ == "__main__":
    image_path = image_path
    predicted_temp, predicted_time = predict_image(image_path)


    print(f"Prediction: Temperature = {predicted_temp}°C, Time = {predicted_time}h")