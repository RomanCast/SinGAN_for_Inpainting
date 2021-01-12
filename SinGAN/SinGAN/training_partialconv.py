'''
TO DO:
- implement create_mask
- where to use partial conv: for G or D or both?
- adapt code with masks
- change loss
'''

import SinGAN.functions as functions
import SinGAN.partialConv2d as PCmodels
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize

import cv2
from SinGAN.patchmatch import contour_holes,is_hole,in_box

def train(opt,Gs,Zs,reals,masks,mask_dir,NoiseAmp):
    print(f"{opt.stop_scale} scales to train")
    real_ = functions.read_image(opt)
    in_s = 0
    scale_num = 0
    real = imresize(real_,opt.scale1,opt)
    reals = functions.creat_reals_pyramid(real,reals,opt)
    if opt.structured:
        # Using a bounding box to locate holes
        masks = create_masks_structured(reals, mask_dir)
    else:
        # Downsampling holes
        masks = create_masks_unstructured(masks, mask_dir, opt)
    for r, m in zip(reals, masks):
        assert r.size() == m.size(), "mask size is different from image size"
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        D_curr,G_curr = init_models(opt)
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals,masks,Gs,Zs,in_s,NoiseAmp,opt)

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(masks,'%s/masks.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return



def train_single_scale(netD,netG,reals,masks,Gs,Zs,in_s,NoiseAmp,opt,centers=None):

    real = reals[len(Gs)]

    mask = masks[len(Gs)]

    assert real.size() == mask.size(), "real and mask do not have the same size"

    n_valid = mask.sum()

    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_mask = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))
    m_mask = nn.ZeroPad2d(int(pad_mask))

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()

            output,_ = netD(real,mask)
            # output = output.to(opt.device)
            #D_real_map = output.detach()
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss(reduction='sum')
                    RMSE = torch.sqrt(criterion(real*mask, z_prev*mask)/n_valid)
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = draw_concat(Gs,Zs,reals,masks,NoiseAmp,in_s,'rand',m_noise,m_image,m_mask,opt)
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs,Zs,reals,masks,NoiseAmp,in_s,'rec',m_noise,m_image,m_mask,opt)
                    criterion = nn.MSELoss(reduction='sum')
                    RMSE = torch.sqrt(criterion(real*mask, z_prev*mask)/n_valid)
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs,Zs,reals,masks,NoiseAmp,in_s,'rand',m_noise,m_image,m_mask,opt)
                prev = m_image(prev)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev

            fake,_ = netG(noise.detach(),prev,m_mask(mask))
            output,_ = netD(fake.detach(),mask)
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = errD_fake.item()

            gradient_penalty = functions.PC_calc_gradient_penalty(netD, real*mask, fake*mask,mask,opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            output,_ = netD(fake,mask)
            #D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss(reduction='sum')
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt+z_prev
                out,_ = netG(Z_opt.detach(),z_prev,m_mask(mask))
                rec_loss = alpha*loss(out*mask,real*mask)/n_valid
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            G_pred,_ = netG(Z_opt.detach(), z_prev,m_mask(mask))
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(G_pred.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt)
    return z_opt,in_s,netG

def draw_concat(Gs,Zs,reals,masks,NoiseAmp,in_s,mode,m_noise,m_image,m_mask,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,mask_curr,noise_amp in zip(Gs,Zs,reals,reals[1:],masks,NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z,_ = G(z_in.detach(),G_z,m_mask(mask_curr))
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,mask_curr,noise_amp in zip(Gs,Zs,reals,reals[1:],masks,NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z,_ = G(z_in.detach(),G_z,m_mask(mask_curr))
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt):

    #generator initialization:
    netG = PCmodels.PCGeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    # netG.apply(PCmodels.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = PCmodels.PCWDiscriminator(opt).to(opt.device)
    # netD.apply(PCmodels.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG

def create_masks_structured(reals,mask_dir):
    '''
    create masks in the case where we have access to the bounding box of the structured hole
    --> mask of downsampled image = reduced bounding box
    --> no downsampling of the mask needed
    '''
    mask_2d =cv2.imread(mask_dir,2)
    boxx,boxy,boxw,boxh = contour_holes(mask_2d,0)
    margin = 1
    im_orig = reals[0]
    h_orig,w_orig = np.size(im_orig,2),np.size(im_orig,3)
    masks = []
    for real in reals:
        h,w = np.size(real,2),np.size(real,3)
        red_h = h_orig/h
        red_w = w_orig/w
        boxx_red = max(0,int(boxx*red_w)-margin)
        boxy_red = max(0,int(boxy*red_h)-margin)
        boxw_red = min(w,int(boxw*red_w)+margin*2)
        boxh_red = min(h,int(boxh*red_h)+margin*2)
        mask = torch.zeros_like(real)
        mask[0,:, boxy_red:boxy_red+boxh_red, boxx_red:boxx_red+boxw_red] = torch.ones((3, boxh_red, boxw_red)).to(mask.device)
        masks.append(mask)
    return masks


def create_masks_unstructured(masks, mask_dir, opt, threshold=0.01):
    '''
    This function downsamples a mask the same way images are downsampled, then uses a threshold to transform that to a binarized mask. Using a threshold sufficiently low ensures that we do not use any pixel from occluded regions. This is done to the mask for training, which is the exact inverse to the mask for inpainting

    inputs:
        mask_dir: string with the location of the mask (in image format)
        opt: arguments
        threshold: (optional) the value defining which pixels of the mask will be kept. The lower the threshold, the more pixels are masked. Empirically, 0.01 seem to be a good threshold value.

    outputs:
        a list of binarized masks
    '''
    mask_orig_ = functions.read_image_dir(mask_dir, opt) # Reading an image that way will move it to GPU

    # Downsample the mask exactly like the image
    mask_orig = imresize(mask_orig_,opt.scale1,opt)
    masks = functions.creat_reals_pyramid(mask_orig, masks, opt)
    new_masks = []
    for mask in masks:
        # Adapt the mask to be in the range [0, 1]
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        # Create a new binarized mask, with 0 and 1 depending on the threshold
        mask_bin = torch.ones_like(mask)
        zeros = torch.tensor([0.]).to(mask_bin.device)
        mask_bin = mask_bin.where(mask > threshold, zeros)
        new_masks.append(mask_bin)
        del mask
    return new_masks
