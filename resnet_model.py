import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetClassifier(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # Load ImageNet-pretrained ResNet18
        weights = ResNet18_Weights.DEFAULT

        self.backbone = resnet18(
            weights=weights
        )

        # Freeze the backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get number of features
        num_features = self.backbone.fc.in_features

        # Replace final classifier
        self.backbone.fc = nn.Sequential(

            nn.Dropout(0.5),

            nn.Linear(
                num_features,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        return self.backbone(x)


if __name__ == "__main__":

    import torch

    model = ResNetClassifier()

    print(model)

    # Test model
    test_input = torch.randn(
        2, 3, 224, 224
    )

    output = model(test_input)

    print()
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)