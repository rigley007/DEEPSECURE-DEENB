import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import os
import config as cfg

# Paths for saving models and adversarial images
models_path = cfg.models_path
adv_img_path = cfg.adv_img_path

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Initialize weights of convolutional layers with a normal distribution
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Initialize BatchNorm weights and biases
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Adversarial Generator class
class Adv_Gen:
    def __init__(self,
                 device,
                 model_extractor,
                 generator,):

        self.device = device
        self.model_extractor = model_extractor
        self.generator = generator
        self.box_min = cfg.BOX_MIN # default: 0
        self.box_max = cfg.BOX_MAX # default: 1
        self.ite = 0
        #self.CELoss = nn.CrossEntropyLoss()

        self.model_extractor.to(device)
        #self.model_extractor.eval()

        self.generator.to(device)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=0.001)

        # Ensure directories for saving models and adversarial images exist
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if not os.path.exists(adv_img_path):
            os.makedirs(adv_img_path)

    def train_batch(self, x):
        self.optimizer_G.zero_grad()  # Reset gradients

        # Generate adversarial images and tagged features
        adv_imgs, tagged_feature = self.generator(x)
        # Extract features from adversarial images using the model
        adv_img_feature = self.model_extractor(adv_imgs)

        # Compute L1 loss between tagged and adversarial image features
        loss_adv = F.l1_loss(tagged_feature, adv_img_feature*0.35)
        loss_adv.backward(retain_graph=True)

        self.optimizer_G.step()

        return loss_adv.item(), adv_imgs

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):
            # Adjust learning rate at specific epochs
            if epoch == 200:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
            if epoch == 400:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
            loss_adv_sum = 0 # Cumulative adversarial loss
            self.ite = epoch
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_adv_batch, adv_img = self.train_batch(images)
                loss_adv_sum += loss_adv_batch


            # print statistics
            # Save adversarial images for visualization
            torchvision.utils.save_image(torch.cat((adv_img[:7], images[:7])),
                                         adv_img_path + str(epoch) + ".png",
                                         normalize=True, scale_each=True, nrow=7)
            num_batch = len(train_dataloader)
            print("epoch %d:\n loss_adv: %.3f, \n" %
                  (epoch, loss_adv_sum/num_batch))
            # Save generator model every 20 epochs
            if epoch%20==0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.generator.state_dict(), netG_file_name)

            print("check")



