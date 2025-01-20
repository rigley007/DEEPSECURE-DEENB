# import torch.nn as nn
# import torchvision.models as pre_models

# # Return first n layers of a pretrained model
# class model_extractor(nn.Module):
#     """
#         Initialize the model extractor.
#         Parameters:
#         - arch (str): The architecture of the pretrained model ('alexnet', 'resnet', 'vgg16', etc.).
#         - num_layers (int): The number of layers to extract from the model.
#         - fix_weights (bool): If True, freeze the weights of the extracted layers to prevent training.
#     """
#     def __init__(self, arch, num_layers, fix_weights):
#         super(model_extractor, self).__init__()
#         # Load the specified pretrained model
#         if arch.startswith('alexnet') :
#             original_model = pre_models.alexnet(pretrained=True)
#         elif arch.startswith('resnet') :
#             original_model = pre_models.resnet18(pretrained=True)
#         elif arch.startswith('vgg16'):
#             original_model = pre_models.vgg16(pretrained=True)
#         else :
#             raise("Not support on this architecture yet")

#         # Extract the first `num_layers` layers from the pretrained model
#         self.features = nn.Sequential(*list(original_model.children())[:num_layers])

#         # Optionally freeze the weights of the extracted layers
#         if fix_weights == True:
#             for p in self.features.parameters():
#                 p.requires_grad = False
#         # Store the name of the architecture for reference
#         self.modelName = arch

#     def forward(self, x):
#         f = self.features(x)
#         return f



import torch.nn as nn
import torchvision.models as pre_models

class model_extractor(nn.Module):
    """A module that extracts and uses specific layers from pretrained models.
    
    This class creates a feature extractor by taking the first n layers from a 
    pretrained model (AlexNet, ResNet18, or VGG16). It can optionally freeze
    the weights of these layers for transfer learning applications.
    
    Parameters:
        arch (str): Architecture name of the pretrained model to use
                   Supported options: 'alexnet', 'resnet', 'vgg16'
        num_layers (int): Number of layers to extract from the beginning of the model
        fix_weights (bool): Whether to freeze the weights of extracted layers
                          True: weights are frozen (useful for transfer learning)
                          False: weights can be updated during training
    
    Raises:
        RuntimeError: If an unsupported architecture is specified
    """
    def __init__(self, arch, num_layers, fix_weights):
        # Initialize parent nn.Module class
        super(model_extractor, self).__init__()
        
        # Load the appropriate pretrained model based on architecture name
        if arch.startswith('alexnet'):
            original_model = pre_models.alexnet(pretrained=True)
        elif arch.startswith('resnet'):
            original_model = pre_models.resnet18(pretrained=True)
        elif arch.startswith('vgg16'):
            original_model = pre_models.vgg16(pretrained=True)
        else:
            raise RuntimeError("Architecture not supported: Only alexnet, resnet, and vgg16 are currently supported")
            
        # Create a new Sequential module with only the first num_layers
        # list(original_model.children()) gets all top-level modules
        # [:num_layers] selects only the first num_layers modules
        # *list unpacks the list into sequential arguments
        self.features = nn.Sequential(*list(original_model.children())[:num_layers])
        
        # If fix_weights is True, freeze all parameters in the extracted layers
        # This prevents their weights from being updated during training
        if fix_weights:
            for p in self.features.parameters():
                p.requires_grad = False
                
        # Store architecture name for reference
        self.modelName = arch
        
    def forward(self, x):
        """Forward pass of the model extractor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Features extracted from the input using the selected layers
        """
        # Pass input through the extracted layers
        f = self.features(x)
        return f
