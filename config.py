# Use CUDA for computation if available
use_cuda = True

# Number of image channels (3 for RGB images)
image_nc = 3

# Number of training epochs
epochs = 80

# Batch size for training
batch_size = 64

# Minimum and maximum values for bounding boxes
BOX_MIN = 0
BOX_MAX = 1

# Pretrained model architecture to use (ResNet-18)
pretrained_model_arch = 'resnet18'

# Number of layers to extract features from
num_layers_ext = 5

# Whether to keep the feature extractor layers fixed during training
ext_fixed = True

# Whether to tag generated images
G_tagged = True

# Size of the tags to be added to generated images
tag_size = 6

# Whether to concatenate generated images with tags
cat_G = True

# Whether to add noise to images
noise_img = True

# Path to the pre-trained generator model
noise_g_path = './models/netG_epoch_80.pth'

# Directory for ImageNet-10 training images
imagenet10_traindir = 'C:/Users/Rui/Pictures/transfer_imgnet_10/train'

# Directory for ImageNet-10 validation images
imagenet10_valdir = 'C:/Users/Rui/Pictures/transfer_imgnet_10/val'

# Path to save models
models_path = './models/'

# Path to save adversarial images
adv_img_path = './images/0526/adv/'
