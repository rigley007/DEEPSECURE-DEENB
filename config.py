BOX_MIN = 0
BOX_MAX = 1
image_nc = 3
num_layers_ext = 5
ext_fixed = True

pretrained_model_arch = 'resnet18'
use_cuda = True
epochs = 100
batch_size = 128

G_tagged = True
tag_size = 6

cat_G = True
noise_img = True
noise_g_path = './models/netG_epoch_100.pth'

imagenet10_traindir = './Pictures/transfer_imgnet_10/train'
imagenet10_valdir = './Pictures/transfer_imgnet_10/val'

models_path = './models/'
adv_img_path = './images/0526/adv/'
