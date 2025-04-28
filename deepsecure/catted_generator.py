
import torch.nn as nn
import torch
from module.resnet_block import ResnetBlock
from module.pre_model_extractor import model_extractor 
import config as cfg

class catted_generator(nn.Module):

    
    """Concatenated Generator that processes and combines features from two inputs.
    
    This generator uses a shared encoder for both inputs and concatenates their
    features before decoding. The decoder architecture adapts based on the depth
    of features extracted from the encoder.
    """
    # def __init__(self, num_encoder_layers, fix_encoder, tagged):
    def __init__(self, num_encoder_layers, fix_encoder, tagged):

        """Initialize the concatenated generator.
        Args:
            num_encoder_layers (int): Number of ResNet layers to use as encoder (5-7)
            fix_encoder (bool): Whether to freeze encoder weights
            tagged (bool): Flag for tagged/marked sample processing
        Raises:

            RuntimeError: If num_encoder_layers < 5 (unsupported configuration)
        """
        super(catted_generator, self).__init__()
        
        # Initialize encoder using ResNet18
        self.encoder = model_extractor('resnet18', num_encoder_layers, fix_encoder)
        self.tagged = tagged
        
        # Check for valid encoder depth
        if num_encoder_layers < 5:
            raise RuntimeError("Encoder depth must be at least 5 layers")
            
        # Configure decoder based on encoder depth
        if num_encoder_layers == 7:
            
            # Decoder for 7-layer encoder (256 input channels): 7 layers?
            decoder_lis = [
                # First block: Process 256-channel features
                ResnetBlock(256),
                ResnetBlock(256),
                
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1, bias=False),
                
                # Second block: Process 128-channel features
                ResnetBlock(128),
                ResnetBlock(128),
                
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
                
                # Third block: Process 64-channel features
                ResnetBlock(64),
                ResnetBlock(64),
                
                nn.UpsamplingNearest2d(scale_factor=2),
                
                # Final output layer to generate RGB image
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()  # Normalize output to [-1, 1]
            ]
            
        elif num_encoder_layers == 6:
            # Decoder network for a 6-layer encoder architecture
            # The decoder progressively upsamples features back to image size
            # Input: 128-channel feature maps from encoder
            # Output: 3-channel RGB image
            # Decoder for 6-layer encoder (128 input channels)
            decoder_lis = [
                
                # First block: Process 128-channel features
                ResnetBlock(128),
                ResnetBlock(128),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
                
                # Second block: Process 64-channel features
                ResnetBlock(64),
                ResnetBlock(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                
                # Final output 
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
            ]
            
        elif num_encoder_layers == 5:
            
            # Decoder for 5-layer encoder (64*2=128 input channels after concatenation)
            
            decoder_lis = [
                # Concatenated features
                ResnetBlock(64*2),
                ResnetBlock(64*2),
                ResnetBlock(64*2),
                nn.UpsamplingNearest2d(scale_factor=2),
                
                # Final output layer
                nn.ConvTranspose2d(64*2, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
            ]
        # Create sequential decoder module
        self.decoder = nn.Sequential(*decoder_lis)
        
    def forward(self, x1, x2):
        """Forward pass through the generator.
        
        Takes two inputs, encodes them separately, concatenates their features,
        and generates the output through the decoder.
        
        Args:
            x1 (torch.Tensor): First input image
            x2 (torch.Tensor): Second input image
            
        Returns:
            tuple:
                - torch.Tensor: Generated output image
                - torch.Tensor: Encoded features of second input
        """
        # Encode both inputs using shared encoder
        x_t_1 = self.encoder(x1)
        x_t_2 = self.encoder(x2)
        
        # Concatenate features along channel dimension and decode
        out = self.decoder(torch.cat((x_t_1, x_t_2), 1))
        # Return both:
    # 1. The generated output image from the decoder
    # 2. The encoded features of the second input (x_t_2) which may be used
    #    for additional loss calculations or downstream tasks
        # Return the generated output image and the encoded features of the second input
        return out, x_t_2 
