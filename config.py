
use_cuda = True # if use GPU
image_nc = 3
epochs = 80 # training epoch
batch_size = 64 # batch size
BOX_MIN = 0 # minimum pixel intensity value
BOX_MAX = 1 # maximum pixel intensity value
pretrained_model_arch = 'resnet18'
num_layers_ext = 5
ext_fixed = True

G_tagged = True
tag_size = 6 # tag size

cat_G = True
noise_img = True
noise_g_path = './models/netG_epoch_80.pth'

imagenet10_traindir = 'C:/Users\Rui\Pictures/transfer_imgnet_10/train'
imagenet10_valdir = 'C:/Users\Rui\Pictures/transfer_imgnet_10/val'

models_path = './models/'
adv_img_path = './images/0526/adv/'
