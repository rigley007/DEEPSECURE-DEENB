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
