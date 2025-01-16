import torch.nn as nn
import torch
from resnet_block import ResnetBlock
from pre_model_extractor import model_extractor
import config as cfg

# Definition of the catted_generator class
class catted_generator(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 fix_encoder,
                 tagged,
                 ):
        """
        Initializes the catted_generator class.

        :param num_encoder_layers: Number of encoder layers to use (affects architecture)
        :param fix_encoder: Boolean indicating whether to freeze encoder layers
        :param tagged: Boolean or additional parameter to include tagged features
        """
        super(catted_generator, self).__init__()

        # Initialize the encoder using a model extractor
        self.encoder = model_extractor('resnet18', num_encoder_layers, fix_encoder)

        self.tagged = tagged
        # Raise an exception for unsupported layer configurations    
        if num_encoder_layers < 5:
            raise("Not support on this layer yet")
        # Define the decoder layers based on the number of encoder layers
        elif num_encoder_layers == 7:
            decoder_lis = [
                ResnetBlock(256),
                ResnetBlock(256),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1, bias=False),
                ResnetBlock(128),
                ResnetBlock(128),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
                ResnetBlock(64),
                ResnetBlock(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
                # state size. image_nc x 224 x 224
            ]
        elif num_encoder_layers == 6:
            decoder_lis = [
                ResnetBlock(128),
                ResnetBlock(128),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
                ResnetBlock(64),
                ResnetBlock(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
                # state size. image_nc x 224 x 224
            ]
        elif num_encoder_layers == 5:
            decoder_lis = [
                ResnetBlock(64*2),
                ResnetBlock(64*2),
                ResnetBlock(64*2),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64*2, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
                # state size. image_nc x 224 x 224
            ]
        # Combine decoder layers into a sequential model
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x1, x2):
        # Extract features from both inputs using the encoder
        x_t_1 = self.encoder(x1)
        x_t_2 = self.encoder(x2)
        # Concatenate encoded features and pass through the decoder
        out = self.decoder(torch.cat((x_t_1, x_t_2),1))

        return out, x_t_2
