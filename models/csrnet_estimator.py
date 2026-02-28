import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import numpy as np
import cv2

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            from torchvision.models import vgg16, VGG16_Weights
            mod = vgg16(weights=VGG16_Weights.DEFAULT)
            self._initialize_weights()
            # Copy specific weights from VGG16
            for i in range(len(self.frontend.state_dict().items())):
                 list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.features.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNetEstimator:
    def __init__(self, model_path=None, use_gpu=True):
        """
        Initialize CSRNet Estimator.
        Args:
            model_path (str): Path to the pre-trained CSRNet model weights.
            use_gpu (bool): Whether to use GPU if available.
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Initializing CSRNet on {self.device}...")

        # If a model_path is given, skip VGG weight copying (load_weights=True)
        # so the network is initialised cleanly before loading the checkpoint.
        use_pretrained_vgg = (model_path is None)
        self.model = CSRNet(load_weights=not use_pretrained_vgg)
        self.model.to(self.device)

        if model_path:
            print(f"Loading CSRNet weights from: {model_path}")
            # weights_only=False needed for older checkpoint formats
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            # Handle checkpoint wrapped in 'state_dict' key or bare OrderedDict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
            print(f"✅ CSRNet weights loaded successfully ({len(state_dict)} layers).")
        else:
            print("⚠️  No weights file provided — using random/VGG init (counts will be inaccurate).")

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def estimate(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to nearest multiple of 8 (important for CSRNet)
        h, w, _ = img.shape
        h = (h // 8) * 8
        w = (w // 8) * 8
        img = cv2.resize(img, (w, h))

        # Apply transform
        img = self.transform(img)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            density_map = self.model(img)

        density_map = density_map.squeeze().cpu().numpy()

        # Final count
        count = float(np.sum(density_map))

        return count, density_map
