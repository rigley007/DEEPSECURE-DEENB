import torch.nn as nn
import torchvision.models as pre_models
from resnet_block import ResnetBlock
from pre_model_extractor import model_extractor

class regular_generator(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 fix_encoder,
                 tagged,
                 ):
        super(regular_generator, self).__init__()

        self.encoder = model_extractor('resnet18', num_encoder_layers, fix_encoder)

        self.tagged = tagged
        if num_encoder_layers < 5:
            raise("Not support on this layer yet")
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
                ResnetBlock(64),
                ResnetBlock(64),
                ResnetBlock(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
                # state size. image_nc x 224 x 224
            ]

        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x_t = self.encoder(x)
        if self.tagged:

            x_t[:, :, :6, :6] = x_t.max()
        x_t = self.decoder(x_t)

        return x_t
