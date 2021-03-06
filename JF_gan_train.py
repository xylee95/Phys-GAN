import os, sys
sys.path.append(os.getcwd())
import time
import functools
import argparse
import numpy as np
import libs as lib
import libs.plot
from tensorboardX import SummaryWriter
from models.wgan import *
from models.checkers import *
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer
from microstructure import MicrostructureDataset
import torch.nn.init as init
import matplotlib.pyplot as plt
import regress
from regress import *

DATA_DIR = '/data/Bernard/DARPA_data/pytorch_normJF_data_train.npy'
VAL_DIR = '/data/Bernard/DARPA_data/pytorch_normJF_data_test.npy'
IMAGE_DATA_SET = 'microstructure'

torch.cuda.set_device(0)    
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

RESTORE_MODE = False  # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0  # starting iteration
OUTPUT_PATH = './output/JFDebugging/'
DIM = 128
CRITIC_ITERS = 5  # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1  # Number of GPUs
BATCH_SIZE = 16  # Batch size. Must be a multiple of N_GPUS
END_ITER = 10000  # How many iterations to train for
LAMBDA = 10  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = DIM * DIM * 6 # Number of pixels in each image
PJ_ITERS = 5
INV_PARAM = 'JF'

def proj_loss(fake_data, real_data, model, real_label):
    """
    Fake data requires to be pushed from tanh range to [0, 1]
    """
    if INV_PARAM == 'JF':
        loss = predict_JF(model, fake_data) - predict_JF(model, real_data)
        #option 1:
        p_loss = torch.norm(loss[:,0]) + torch.norm(loss[:,1])
        #option 2:
        #p_loss =  torch.norm(loss)
        return p_loss
    elif INV_PARAM == 'J':
        p_loss = torch.norm(predict_J(model, fake_data) - predict_J(model, real_data))
        return p_loss

def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def load_data(path_to_folder, train):
    data_transform = transforms.Compose([
        # transforms.Scale(64),
        # transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if IMAGE_DATA_SET == 'microstructure':
        dataset = MicrostructureDataset(path_to_folder, mode=INV_PARAM)
    else:
        dataset = datasets.ImageFolder(root=path_to_folder, transform=data_transform)
    
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                                                 pin_memory=True)
    return dataset_loader

def training_data_loader():
    return load_data(DATA_DIR, train='train')

def val_data_loader():
    return load_data(VAL_DIR, train='valid')

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()

    alpha = alpha.view(BATCH_SIZE, CATEGORY, DIM, DIM)              # Changed the CATEGORY from 1
    alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, CATEGORY, DIM, DIM)      # Changed the CATEGORY from 1
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_image(netG, noise=None, lv=None):
    if noise is None:
        noise = gen_rand_noise()
    if lv is None:
       # lv = torch.randn(BATCH_SIZE, 1)
       lv = torch.rand(BATCH_SIZE, 2)
       summ = torch.sum(lv, dim=1).unsqueeze(1)
       lv = lv /summ
       lv = lv.to(device)
    with torch.no_grad():
        noisev = noise
        lv_v = lv
    samples = netG(noisev, lv_v)
    samples = torch.argmax(samples.view(BATCH_SIZE, CATEGORY, DIM, DIM), dim=1).unsqueeze(1)
    samples = (samples * 255/CATEGORY)
    samples = samples.int()
    return samples

def gen_rand_noise(): # z
    noise = torch.randn(BATCH_SIZE, 127) #reduce from 128 to 127 to account for additonal property
    noise = noise.to(device)
    return noise

def predict_JF(model,x):
    model.eval()
    JF = model(x)
    return JF

# Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    print('Loading surrogate model weights')

    #------Load regressor as invariant checker--------#
    regressor = regress.JF_Net()
    regressor.load_state_dict(torch.load('JF_regressor.pt', map_location='cuda:0'))
    regressor.eval()
    regressor.to(device)
    for params in regressor.parameters(): #Freeze surrogate
        params.requires_grad_(False)

    #-------Load training data---------------------
    print("Loading the Training Data")
    dataloader = training_data_loader()
    dataiter = iter(dataloader)

    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
        print("-------------------------")
        print("Iter: " + str(iteration))
        print("-------------------------")
        start = timer()
        # ---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        try:
            real_data, real_J, real_ff = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            real_data, real_J, real_ff = dataiter.next()

        if INV_PARAM == 'JF':
            real_data = real_data.unsqueeze(1) #batch, 1, 128, 128
        elif INV_PARAM == 'J':
            real_data = real_data.unsqueeze(1) #batch, 1, 128, 128

        real_p = regressor(real_data.to(device))
        real_p = real_p.to(device)

        for i in range(GENER_ITERS):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise()    #generate random z vector (batch,128)
            noise.requires_grad_(True)

            #z (batch,127), real_p (batch,2), making it (batch,129)
            fake_data = aG(noise, real_p)
            gen_cost = aD(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost = -gen_cost
            gen_cost.backward()

        optimizer_g.step()
        end = timer()

        print(f'---train G elapsed time: {end - start}')
        print('Fake Min:', fake_data.min(), 'Real Min:',real_data.min())
        print('Fake Max:', fake_data.max(), 'Real Max:',real_data.max())
        
        # Projection steps: ensures invariance
        pj_cost = None
        for i in range(PJ_ITERS):
            print('Projection iters: {}'.format(i))
            aG.zero_grad()
            noise = gen_rand_noise()
            noise.requires_grad=True
            fake_data = aG(noise, real_p)
            pj_cost = proj_loss(fake_data.view(-1, CATEGORY, DIM, DIM), real_data.to(device), regressor.to(device), real_p)
            pj_cost = pj_cost.mean()
            pj_cost.backward()
            optimizer_pj.step()

        # ---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            print("Critic iter: " + str(i))

            start = timer()
            aD.zero_grad()

            # gen fake data and load real data
            noise = gen_rand_noise()
            try:
                batch, real_J, real_ff = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                batch, real_J, real_ff = dataiter.next()

            # batch = batch[0] #batch[1] contains labels
            real_data = batch.to(device=device, dtype=torch.float)  # TODO: modify load_data for each loading
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D

                if INV_PARAM == 'JF':
                    real_data = real_data.unsqueeze(1)
                elif INV_PARAM == 'J':
                    real_data = real_data.unsqueeze(1)

                real_p = regressor(real_data.to(device))
                real_p = real_p.to(device)

            end = timer();
            print(f'---gen G elapsed time: {end-start}')
            start = timer()
            fake_data = aG(noisev, real_p).detach()
            #fake_data = aG(noisev, label).detach()
            end = timer();
            print(f'---load real imgs elapsed time: {end-start}')
            start = timer()

            # train with real data
            disc_real = aD(real_data)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = aD(fake_data)
            disc_fake = disc_fake.mean()
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake - disc_real

            optimizer_d.step()
            # ------------------VISUALIZATION----------
            if i == CRITIC_ITERS - 1:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                writer.add_scalar('data/disc_fake', disc_fake, iteration)
                writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)

            end = timer();
            print(f'---train D elapsed time: {end-start}')
        # ---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)

        lib.plot.plot(OUTPUT_PATH + 'time', time.time() - start_time)
        lib.plot.plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())
        if iteration % 50 == 0:
            fake_2 = torch.argmax(fake_data.view(BATCH_SIZE, CATEGORY, DIM, DIM), dim = 1).unsqueeze(1)
            fake_2 = (fake_2 * 255/6)
            fake_2 = fake_2.int()
            fake_2 = fake_2.cpu().detach().clone()
            # fake_2 = (fake_2 + 1.0)/2.0
            fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
            writer.add_image('G/images', fake_2, iteration)
        if iteration % 10 == 0:
            val_loader = val_data_loader()
            # p2_vals = []
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                # print(images[0])
                # print(images[0].shape)
                # imgs = torch.FloatTensor(np.float32(images))
                imgs = torch.FloatTensor(np.float32(images[0]))
                # print(imgs.size())
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs
                # Sample random p2's for analysis
                rn = np.random.rand()
                # if rn > 0.1 and len(p2_vals) < 64:
                #    p2_vals.append(p2_fn(imgs.unsqueeze(0)))
                D = aD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(OUTPUT_PATH + 'dev_disc_cost.png', np.mean(dev_disc_costs))
            lib.plot.flush()
            # p2_vals = torch.stack(p2_vals, dim=0).squeeze(1).to(device)
            # if p2_vals.size()[0] != BATCH_SIZE:
            #    continue
            gen_images = generate_image(aG, fixed_noise)
            # torchvision.utils.save_image(gen_images, OUTPUT_PATH + 'samples_{}.png'.format(iteration), nrow=8,
            #                              padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
            # ----------------------Save model----------------------
            torch.save(aG, OUTPUT_PATH + "generator.pt")
            torch.save(aD, OUTPUT_PATH + "discriminator.pt")
        lib.plot.tick()

if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    fixed_noise = gen_rand_noise()

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    if RESTORE_MODE:
        aG = torch.load(OUTPUT_PATH + "generator.pt")
        aD = torch.load(OUTPUT_PATH + "discriminator.pt")
    else:
        aG = GoodGenerator(dim=128, output_dim = DIM * DIM * 6, ctrl_dim=CATEGORY)
        aD = GoodDiscriminator(dim=128)
        #initilize gen and disc weights
        aG.apply(weights_init)
        aD.apply(weights_init)

    LR = 1e-5
    optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0, 0.9))  # Gen loss
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0, 0.9))  # Disc loss
    optimizer_pj = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0, 0.9)) # Projection Loss

    aG = aG.to(device)
    aD = aD.to(device)
    writer = SummaryWriter()
    train()


