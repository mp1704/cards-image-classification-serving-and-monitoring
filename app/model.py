import timm
from torch import nn


class MyModel(nn.Module):
    def __init__(self, num_classes=53):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        # remove the classifier layer of efficientnet_b0
        self.feature_extractor = nn.Sequential(*list(self.base_model.children()))[:-1]
        self.classifier = nn.Linear(
            in_features=1280, out_features=num_classes
        )  # new head of model

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)  # (batch_size, num_classes)
