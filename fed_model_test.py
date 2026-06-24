import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 352

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CamouflageSegNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = timm.create_model(
            "pvt_v2_b2",
            pretrained=False,
            features_only=True,
            out_indices=(0,1,2,3),
        )

        enc_channels = [64,128,320,512]

        self.lat4 = ConvBnRelu(enc_channels[3],256)
        self.lat3 = ConvBnRelu(enc_channels[2],256)
        self.lat2 = ConvBnRelu(enc_channels[1],128)
        self.lat1 = ConvBnRelu(enc_channels[0],64)

        self.merge3 = ConvBnRelu(512,256)
        self.merge2 = ConvBnRelu(384,128)
        self.merge1 = ConvBnRelu(192,64)

        self.seg_head = nn.Sequential(
            ConvBnRelu(64,64),
            nn.Conv2d(64,1,1)
        )

        self.edge_head = nn.Sequential(
            ConvBnRelu(64,32),
            nn.Conv2d(32,1,1)
        )

    def forward(self,x):

        H,W = x.shape[2:]

        f1,f2,f3,f4 = self.encoder(x)

        p4 = self.lat4(f4)
        p3 = self.lat3(f3)
        p2 = self.lat2(f2)
        p1 = self.lat1(f1)

        p3 = self.merge3(torch.cat([
            F.interpolate(
                p4,
                size=p3.shape[2:],
                mode="bilinear",
                align_corners=False
            ),
            p3
        ],dim=1))

        p2 = self.merge2(torch.cat([
            F.interpolate(
                p3,
                size=p2.shape[2:],
                mode="bilinear",
                align_corners=False
            ),
            p2
        ],dim=1))

        p1 = self.merge1(torch.cat([
            F.interpolate(
                p2,
                size=p1.shape[2:],
                mode="bilinear",
                align_corners=False
            ),
            p1
        ],dim=1))

        out = F.interpolate(
            p1,
            size=(H,W),
            mode="bilinear",
            align_corners=False
        )

        seg = self.seg_head(out)
        edge = self.edge_head(out)

        return seg, edge, p4


def preprocess(img_path):

    img = Image.open(img_path).convert("RGB")
    original = np.array(img)

    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    img = np.array(img).astype(np.float32) / 255.0

    img = (img - MEAN) / STD

    img = img.transpose(2,0,1)

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor, original


MODEL_PATH = "fl_checkpoints/global_best.pth"

model = CamouflageSegNet().to(DEVICE)

ckpt = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    ckpt["model_state"]
)

model.eval()

print("Model loaded successfully")



IMAGE_PATH = "test.jpeg"

x, original = preprocess(IMAGE_PATH)

with torch.no_grad():

    x = x.to(DEVICE)

    pred, _, _ = model(x)

    pred = torch.sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()

mask = (pred > 0.5).astype(np.uint8)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(original)
plt.title("Input")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(pred)
plt.title("Probability Map")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(mask)
plt.title("Binary Mask")
plt.axis("off")

plt.show()