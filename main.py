import torch
import config as cfg
from imagenet10_dataloader import get_data_loaders
from adv_image import Adv_Gen
from cat_adv_image import Cat_Adv_Gen
from regular_generator import regular_generator
from catted_generator import catted_generator
from pre_model_extractor import model_extractor

if __name__ == '__main__':

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")

    train_loader, val_loader = get_data_loaders()

    feature_ext = model_extractor(cfg.pretrained_model_arch, cfg.num_layers_ext, cfg.ext_fixed)

    if cfg.cat_G:
        if cfg.noise_img:
            reg_generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            reg_generator.load_state_dict(torch.load(cfg.noise_g_path))
            reg_generator.eval()
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            advGen = Cat_Adv_Gen(device, feature_ext, generator, reg_generator)
        else:
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            advGen = Cat_Adv_Gen(device, feature_ext, generator, False)
    else:
        generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
        advGen = Adv_Gen(device, feature_ext, generator)

    advGen.train(train_loader, cfg.epochs)

# Import required libraries
import torch
import config as cfg  # Configuration settings
from imagenet10_dataloader import get_data_loaders  # Custom data loading utility
from adv_image import Adv_Gen  # Regular adversarial image generator
from cat_adv_image import Cat_Adv_Gen  # Concatenated adversarial image generator
from regular_generator import regular_generator  # Standard generator architecture
from catted_generator import catted_generator  # Concatenated generator architecture
from pre_model_extractor import model_extractor  # Feature extraction model

if __name__ == '__main__':
    # Check and print CUDA availability for GPU acceleration
    print("CUDA Available: ", torch.cuda.is_available())
    
    # Set the device (GPU if available and enabled in config, otherwise CPU)
    device = torch.device("cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")
    
    # Load training and validation data
    train_loader, val_loader = get_data_loaders()
    
    # Initialize the feature extractor with specified architecture and settings
    feature_ext = model_extractor(
        cfg.pretrained_model_arch,  # Architecture type (e.g., ResNet, VGG)
        cfg.num_layers_ext,         # Number of layers to use for extraction
        cfg.ext_fixed              # Whether to freeze the extractor weights
    )
    
    # Conditional branch for concatenated generator setup
    if cfg.cat_G:
        # Sub-branch for noise image incorporation
        if cfg.noise_img:
            # Load pre-trained regular generator for noise generation
            reg_generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            reg_generator.load_state_dict(torch.load(cfg.noise_g_path))
            reg_generator.eval()  # Set to evaluation mode
            
            # Initialize concatenated generator
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            # Create adversarial generator with noise incorporation
            advGen = Cat_Adv_Gen(device, feature_ext, generator, reg_generator)
        else:
            # Initialize concatenated generator without noise
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            advGen = Cat_Adv_Gen(device, feature_ext, generator, False)
    else:
        # Initialize regular generator setup
        generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
        advGen = Adv_Gen(device, feature_ext, generator)
    
    # Start the training process with specified number of epochs
    advGen.train(train_loader, cfg.epochs)
