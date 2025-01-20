import torch.nn as nn
import torch
from module.resnet_block import ResnetBlock  # Import custom ResNet block
from module.pre_model_extractor import model_extractor  # Import pre-trained model extractor
import config as cfg  # Import configuration

class catted_generator(nn.Module):
    """
    A generator model using ResNet as an encoder.
    It takes two input images, extracts features, concatenates them,
    and reconstructs an output image using a decoder.
    """

    def __init__(self, num_encoder_layers, fix_encoder, tagged):
        super(catted_generator, self).__init__()

        # Initialize encoder using ResNet-18
        self.encoder = model_extractor('resnet18', num_encoder_layers, fix_encoder)

        self.tagged = tagged
        if num_encoder_layers < 5:
            raise ValueError("Not supported for layers less than 5")
        
        # Define decoder based on encoder depth
        elif num_encoder_layers == 7:
            decoder_lis = [
                ResnetBlock(256), ResnetBlock(256),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1, bias=False),
                ResnetBlock(128), ResnetBlock(128),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
                ResnetBlock(64), ResnetBlock(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()  # Output normalized to [-1, 1]
            ]
        elif num_encoder_layers == 6:
            decoder_lis = [
                ResnetBlock(128), ResnetBlock(128),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
                ResnetBlock(64), ResnetBlock(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
            ]
        elif num_encoder_layers == 5:
            decoder_lis = [
                ResnetBlock(64*2), ResnetBlock(64*2), ResnetBlock(64*2),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64*2, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
            ]

        # Construct the decoder
        self.decoder = nn.Sequential(*decoder_lis)
        

    def forward(self, x1, x2):
        """
        Forward pass:
        1. Encode both input images separately.
        2. Concatenate encoded features.
        3. Decode to reconstruct an output image.
        """
        x_t_1 = self.encoder(x1)  # Encode first image
        x_t_2 = self.encoder(x2)  # Encode second image
        out = self.decoder(torch.cat((x_t_1, x_t_2), 1))  # Concatenate and decode

        return out, x_t_2  # Return generated output and second image features
