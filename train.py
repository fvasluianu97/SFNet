import argparse
import os
import torch
import torchvision.transforms as transforms
from dataloader import ImageSet
from dataloader_f7k import Flare_Image_Loader, Image_Pair_Loader
from spfnet import SPFNet
from loss import compute_loss, custom_mipi_loss
from perceptual_loss  import PerceptualLossModule
from restormer import Restormer
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from uformer import Uformer
from utils import validate_model, save_checkpoint, load_checkpoint
from PIL import Image

import wandb
wandb.init(project="spfnet-devel-phase-f7kpp")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
    parser.add_argument("--decay_epoch", type=int, default=200, help="epoch from which to start lr decay")
    parser.add_argument("--n_steps", type=int, default=5, help="number of step decays for the learning rate")

    parser.add_argument("--data_src", type=int, default=2, help="[0] DSLR flare [1] nighttime flare [2] mipi24")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")

    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    
    parser.add_argument("--alpha_1", type=float, default=0.7, help="weight for SSIM loss")
    parser.add_argument("--alpha_2", type=float, default=0.0, help="weight for gradient loss")
    parser.add_argument("--alpha_3", type=float, default=0.8, help="weight for content loss")

    parser.add_argument("--img_height", type=int, default=640, help="size of image height")
    parser.add_argument("--img_width", type=int, default=640, help="size of image width")
    parser.add_argument("--valid_checkpoint", type=int, default=1, help="checkpoint for visual inspection")
    parser.add_argument("--save_checkpoint", type=int, default=10, help="Save checkpoint interval")
    parser.add_argument("--description", default="spfnet-direct-transfer-mipi24-alexssimloss", help="Experiment name")
    parser.add_argument("--load", type=int, default=1, help="Load the best checkpoint from disk")

    opt = parser.parse_args()
    print(opt)


    if torch.cuda.is_available():
        print("Using {} CUDA devices available".format(torch.cuda.device_count()))
        model_net = torch.nn.DataParallel(SPFNet(16, device="cuda"))
        # model_net = torch.nn.DataParallel(Restormer())
        # input_size = 512
        # depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        # model_net = torch.nn.DataParallel(Uformer(img_size=input_size, embed_dim=16, depths=depths,
        #                              win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff',
        #                              modulator=True,
        #                              shift_flag=False))
    else:
        model_net = SPFNet(16, device="cpu")

    betas = (opt.b1, opt.b2)
    optimizer = torch.optim.Adam(model_net.parameters(), lr=opt.lr, betas=betas)

    step_sz = (opt.n_epochs - opt.decay_epoch) // opt.n_steps
    milestones = [opt.decay_epoch + k * step_sz for k in range(opt.n_steps)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.6)

    if opt.data_src == 0:
        train_dataloader = DataLoader(
            ImageSet("data/flare-removal", "train"), batch_size=opt.batch_size,
            shuffle=True, num_workers=opt.n_cpu)

        val_dataloader = DataLoader(
            ImageSet("data/flare-removal", "validation", aug=False), batch_size=1,
            shuffle=False, num_workers=opt.n_cpu)
    elif opt.data_src == 1:
        transform_base = {
            'img_size': 512,
        }
        transform_flare = {
            'scale_min': 0.7,
            'scale_max': 1.2,
            'translate': 100,
            'shear': 20
        }
        
        flare_image_set = Flare_Image_Loader('data/Flickr24K', transform_base, transform_flare)
        flare_image_set.load_scattering_flare('Flare7k_scattering',
                                            'data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare')
        flare_image_set.load_scattering_flare('Real_scattering1',
                                              'data/Flare7Kpp/Flare-R/Compound_Flare')

        flare_image_set.load_reflective_flare('Flare7k_reflective',
                                              'data/Flare7Kpp/Flare7K/Reflective_Flare')
        flare_image_set.load_reflective_flare('Real_reflective1',None)
    
        flare_image_set.load_light_source("Flare7k_light",
                                          "data/Flare7Kpp/Flare7K/Scattering_Flare/Light_Source")
        flare_image_set.load_light_source("Real_light1",
                                          "data/Flare7Kpp/Flare-R/Light_Source")

        train_dataloader = DataLoader(flare_image_set, batch_size=opt.batch_size,  shuffle=True, num_workers=opt.n_cpu)

        val_dataset = Image_Pair_Loader("data/Flare7Kpp/test_data/real/gt",
                              "data/Flare7Kpp/test_data/real/input")
        val_dataset.load_additional_samples("data/Flare7Kpp/test_data/synthetic/gt",
                                            "data/Flare7Kpp/test_data/synthetic/input")
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.n_cpu)
        print("Validation using {} samples".format(len(val_dataloader)))
    else:
        train_dataloader = DataLoader(
            ImageSet("data/mipi24", "train", size=(opt.img_width, opt.img_height)), batch_size=opt.batch_size,
            shuffle=True, num_workers=opt.n_cpu)

        val_dataset = Image_Pair_Loader("data/Flare7Kpp/test_data/real/gt",
                                        "data/Flare7Kpp/test_data/real/input")
        val_dataset.load_additional_samples("data/Flare7Kpp/test_data/synthetic/gt",
                                            "data/Flare7Kpp/test_data/synthetic/input")
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.n_cpu)
        print("Validation using {} samples".format(len(val_dataloader)))



    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
        model_net = model_net.cuda()
        field_loss_module = PerceptualLossModule(device=torch.device("cuda"))
    else:
        Tensor = torch.FloatTensor
        field_loss_module = PerceptualLossModule(device=torch.device("cpu"))

    
    

    best_psnr = 0
    bst_ckp_path = "checkpoints/{}/best".format(opt.description)
    
    if opt.load == 1:
        load_model_checkpoint = "checkpoints/{}/best/checkpoint.pt".format(opt.description)
        model_net, optimizer, scheduler = load_checkpoint(load_model_checkpoint, model_net, optimizer, scheduler)

        out_path = "output/{}/best_start".format(opt.description)
        os.makedirs(out_path, exist_ok=True)
        val_report = validate_model(model_net, val_dataloader, save_disk=True, out_dir=out_path)

        wandb.log({
            "val_mse": val_report["MSE"],
            "val_psnr": val_report["PSNR"],
            "val_ssim": val_report["SSIM"],
        })

        # best_psnr = val_report["PSNR"]
        print("Starting at - validation MSE: {:.3f} - PSNR: {:.3f} - SSIM: {:.4f}".format(val_report["MSE"],
                                                                                          val_report["PSNR"],
                                                                                          val_report["SSIM"]))
    else:
        os.makedirs(bst_ckp_path, exist_ok=True)

    for epoch in range(opt.n_epochs):
        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            input_img = Variable(batch[0].type(Tensor))
            gt_img = Variable(batch[1].type(Tensor))
            optimizer.zero_grad()
            out_img = model_net(input_img)
            loss = custom_mipi_loss(out_img, gt_img, opt) #compute_loss(out_img, gt_img, opt, field_loss_module=field_loss_module)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_net.parameters(), 0.01)
            optimizer.step()
            epoch_loss += loss.detach().item()

        scheduler.step()
        print("Epoch: {} - training_loss {:.4f}".format(epoch + 1, epoch_loss))

        if (epoch + 1) % opt.valid_checkpoint == 0:
            save_disk = (epoch + 1) % opt.save_checkpoint == 0

            if save_disk:
                out_path = "output/{}/{}".format(opt.description, epoch + 1)
                ckp_path = "checkpoints/{}/{}".format(opt.description, epoch + 1)

                os.makedirs(out_path, exist_ok=True)
                os.makedirs(ckp_path, exist_ok=True)
                save_checkpoint(ckp_path, model_net, optimizer, scheduler)
            else:
                out_path = None

            val_report = validate_model(model_net, val_dataloader, save_disk=save_disk, out_dir=out_path)
            torch.cuda.empty_cache()

            wandb.log({
                "val_mse": val_report["MSE"],
                "val_psnr": val_report["PSNR"],
                "val_ssim": val_report["SSIM"],
            })

            
            if val_report["PSNR"] > best_psnr and epoch > 5:
                best_psnr = val_report["PSNR"]
                save_checkpoint(bst_ckp_path, model_net, optimizer, scheduler)

            print("Epoch: {} - validation MSE: {:.3f} - PSNR: {:.3f} - SSIM: {:.4f}".format(epoch + 1, val_report["MSE"],
                                                           val_report["PSNR"], val_report["SSIM"]))




