import torch
import config as cfg
from imagenet10_dataloader import get_data_loaders
from adv_image import Adv_Gen
from cat_adv_image import Cat_Adv_Gen
from regular_generator import regular_generator
from catted_generator import catted_generator
from pre_model_extractor import model_extractor

# Main entry point of the script
if __name__ == '__main__':
    # Check if CUDA is available and print the result
    print("CUDA Available: ", torch.cuda.is_available())

    # Set the device to CUDA if available and configured to use, otherwise use CPU
    device = torch.device("cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")  # Output the chosen device

    # Get the data loaders for training and validation datasets
    train_loader, val_loader = get_data_loaders()
    print("Data loaders loaded. Train set size:", len(train_loader.dataset), "Validation set size:", len(val_loader.dataset))

    # Extract features using a pre-trained model
    print(f"Initializing model extractor with {cfg.pretrained_model_arch} architecture and extracting {cfg.num_layers_ext} layers.")
    feature_ext = model_extractor(cfg.pretrained_model_arch, cfg.num_layers_ext, cfg.ext_fixed)
    
    # Check if concatenated generator is to be used
    if cfg.cat_G:
        print("Using concatenated generator architecture.")
        if cfg.noise_img:
            # Initialize and load the regular generator with noise images
            print(f"Loading regular generator from {cfg.noise_g_path}")
            reg_generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            reg_generator.load_state_dict(torch.load(cfg.noise_g_path))  # Load pre-trained weights
            reg_generator.eval()  # Set generator to evaluation mode
            print("Regular generator loaded and set to eval mode.")
            
            # Initialize the concatenated generator
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            print("Concatenated generator initialized.")
            
            # Create an instance of Cat_Adv_Gen with both generators
            advGen = Cat_Adv_Gen(device, feature_ext, generator, reg_generator)
            print("Cat_Adv_Gen instance created with both generators.")
        else:
            # Initialize the concatenated generator without noise images
            print("Loading concatenated generator without noise images.")
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            print("Concatenated generator initialized.")
            
            # Create an instance of Cat_Adv_Gen with only the concatenated generator
            advGen = Cat_Adv_Gen(device, feature_ext, generator, False)
            print("Cat_Adv_Gen instance created without regular generator.")
    else:
        # Initialize the regular generator if concatenated generator is not used
        print("Using regular generator architecture.")
        
        # Create adversarial generator with concatenated architecture (False means no regular generator)
        generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
        print("Regular generator initialized.")
        
        # Create an instance of Adv_Gen with the regular generator
        advGen = Adv_Gen(device, feature_ext, generator)
        print("Adv_Gen instance created with regular generator.")

    # Print the architecture of the generator
    print(f"Generator architecture: {generator}")
