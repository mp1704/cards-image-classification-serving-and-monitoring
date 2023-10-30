
from torch import nn
import timm

class MyModel(nn.Module):
    def __init__(self, num_classes = 53):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained = True)
        self.feature_extractor = nn.Sequential(*list(self.base_model.children()))[:-1] # output = 1280
        self.classifier = nn.Linear(in_features = 1280, out_features = num_classes) # output = num_classes

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x) # (32, 53) ~ (batch_size, num_classes) 
