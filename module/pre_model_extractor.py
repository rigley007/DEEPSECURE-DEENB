import torch.nn as nn
import torchvision.models as pre_models

# Return first n layers of a pretrained model
class model_extractor(nn.Module):
    def __init__(self, arch, num_layers, fix_weights):
        super(model_extractor, self).__init__()
        if arch.startswith('alexnet') :
            original_model = pre_models.alexnet(pretrained=True)
        elif arch.startswith('resnet') :
            original_model = pre_models.resnet18(pretrained=True)
        elif arch.startswith('vgg16'):
            original_model = pre_models.vgg16(pretrained=True)
        else :
            raise("Not support on this architecture yet")

        self.features = nn.Sequential(*list(original_model.children())[:num_layers])
        if fix_weights == True:
            for p in self.features.parameters():
                p.requires_grad = False
        self.modelName = arch

    def forward(self, x):
        f = self.features(x)
        return f
