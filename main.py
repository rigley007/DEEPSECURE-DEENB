#import added
import torch
import config as cfg
from imagenet10_dataloader import get_data_loaders
from adv_image import Adv_Gen
from cat_adv_image import Cat_Adv_Gen
from regular_generator import regular_generator
from catted_generator import catted_generator
from pre_model_extractor import model_extractor

if __name__ == '__main__':
    
    print("CUDA Available: ", torch.cuda.is_available())
    logging.info("CUDA Available: %s", torch.cuda.is_available())
    device = torch.device("cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")
    if cfg.use_cuda and not torch.cuda.is_available():
        logging.warning("CUDA is enabled in the configuration, but no CUDA devices are available. Falling back to CPU.")
    else:
        logging.info("Using device: %s", device)
    # Set the device to CUDA if available and configured to use, otherwise use CPU
    device = torch.device("cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")
    # Load training and validation data using custom data loaders
    # These contain the ImageNet10 dataset split into training and validation sets
    # Get the data loaders for training and validation datasets
    train_loader, val_loader = get_data_loaders()
    # Initialize the feature extractor using a pre-trained model
    # This will extract intermediate features from images for adversarial generation
    # Parameters:
    #   - pretrained_model_arch: Architecture type (e.g., ResNet, VGG)
    #   - num_layers_ext: Number of layers to extract features from
    #   - ext_fixed: Whether to freeze the extractor weights
    # Extract features using a pre-trained model
    feature_ext = model_extractor(cfg.pretrained_model_arch, cfg.num_layers_ext, cfg.ext_fixed)

    # Check if concatenated generator is to be used

    if cfg.cat_G:# If using concatenated generator architecture

        if cfg.noise_img:

            # Load pre-trained regular generator for noise handling

            # Initialize and load the regular generator with noise images
            reg_generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            reg_generator.load_state_dict(torch.load(cfg.noise_g_path))
            reg_generator.eval()
            
            # Initialize the concatenated generator
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            # Create an instance of Cat_Adv_Gen with both generators
            advGen = Cat_Adv_Gen(device, feature_ext, generator, reg_generator)
            
        else:
            # Initialize the concatenated generator without noise images
            # 
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            advGen = Cat_Adv_Gen(device, feature_ext, generator, False)
    else:
        # Initialize the regular generator
        # Create adversarial generator with concatenated architecture
        # False parameter indicates no regular generator for noise
        generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
        # Create an instance of Adv_Gen with the regular generator
        advGen = Adv_Gen(device, feature_ext, generator)



        # Start training process for specified number of epochs
    # This will train the generator to create adversarial images
    advGen.train(train_loader, cfg.epochs)

