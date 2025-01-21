import argparse
import os
import torch
from dwnet import fusion_net as DWTGen
from model_convnext import fusion_net as DWTFCCGen
from dataloader import ImageSet
from dconv_model import DistillNet
from gradloader import GradualImageSet
from loss import compute_loss
from perceptual_loss import PerceptualLossModule
from restormer import Restormer
from spfnet import SPFNet
from swinir import SwinIR
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from uformer import Uformer
from utils import compute_maxchann_map
from utils import validate_model, save_checkpoint, load_checkpoint

import wandb
wandb.init(project="spfnet-devel-phase")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_rounds", type=int, default=400, help="number of epochs of training")
    parser.add_argument("--past_rounds", type=int, default=0, help="number of epochs of training")
    parser.add_argument("--round_samples", type=int, default=1000, help="number of samples per round")
    parser.add_argument("--decay_round", type=int, default=250, help="Round from which to start lr decay")
    parser.add_argument("--decay_steps", type=int, default=2, help="number of step decays for the learning rate")

    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    
    parser.add_argument("--alpha_1", type=float, default=0.7, help="weight for SSIM loss")
    parser.add_argument("--alpha_2", type=float, default=0.5, help="weight for gradient loss")
    parser.add_argument("--alpha_3", type=float, default=0.8, help="weight for content loss")

    parser.add_argument("--img_h", type=int, default=256, help="size of image height")
    parser.add_argument("--img_w", type=int, default=256, help="size of image width")
    parser.add_argument("--valid_checkpoint", type=int, default=1, help="checkpoint for visual inspection")
    parser.add_argument("--save_checkpoint", type=int, default=25, help="Save checkpoint interval")
    parser.add_argument("--description", default="restormer_gradual", help="Experiment name")
    parser.add_argument("--load", type=int, default=0, help="Load the best checkpoint from disk")

    opt = parser.parse_args()
    print(opt)
    if torch.cuda.is_available():
        print("Using {} CUDA devices available".format(torch.cuda.device_count()))
        device = "cuda"
    else:
        device = "cpu"


    if torch.cuda.is_available():
        # model_net = torch.nn.DataParallel(Restormer())
        model_net = torch.nn.DataParallel(SPFNet(16, device=device))
        # model_net = torch.nn.DataParallel(DistillNet(num_iblocks=6, num_ops=4, device="cuda"), device_ids=[0,3])
        # model_net =  torch.nn.DataParallel(DWTFCCGen())
        # input_size = 512
        # depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        # model_net = torch.nn.DataParallel(Uformer(img_size=input_size, embed_dim=16, depths=depths,
        #                             win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff',
        #                             modulator=True,
        #                             shift_flag=False))
        window_size = 8
        height = 512
        width = 512

        # model_net = SwinIR(upscale=2, img_size=(height, width),
        #                window_size=window_size, img_range=1., depths=[3, 3, 3, 3],
        #                embed_dim=60, num_heads=[4, 4, 4, 4], mlp_ratio=2, upsampler='none')

        Tensor = torch.cuda.FloatTensor
        field_loss_module = PerceptualLossModule(device=torch.device("cuda"))
        model_net = model_net.cuda()
    else:
        model_net = DistillNet(num_iblocks=6, num_ops=4, device="cpu")
        Tensor = torch.FloatTensor
        field_loss_module = PerceptualLossModule(device=torch.device("cpu"))

    betas = (opt.b1, opt.b2)
    optimizer = torch.optim.Adam(model_net.parameters(), lr=opt.lr, betas=betas)

    step_sz = (opt.num_rounds - opt.decay_round) // opt.decay_steps
    milestones = [opt.decay_round + k * step_sz for k in range(opt.decay_steps)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    if opt.load == 1:
        load_model_checkpoint = "/local/checkpoints/{}/best/checkpoint.pt".format(opt.description)
        model_net, optimizer, scheduler = load_checkpoint(load_model_checkpoint, model_net, optimizer, scheduler)


    startidx = opt.past_rounds * opt.round_samples * opt.batch_size
    numsamples = opt.num_rounds * opt.round_samples * opt.batch_size - startidx

    train_dataloader = DataLoader(
        GradualImageSet("/local/data/LF_patterns", numsamples=numsamples, startidx=startidx, size=(opt.img_h, opt.img_w), aug=True, mode="resize"),
        batch_size=opt.batch_size, shuffle=False, drop_last=True, num_workers=opt.n_cpu)

    val_dataloader = DataLoader(
        ImageSet("/local/data/flare-removal", "validation", size=(512, 512), aug=False), batch_size=1,
        shuffle=False, num_workers=opt.n_cpu)

    best_psnr = 0
    bst_ckp_path = "/local/checkpoints/{}/best".format(opt.description)

    if opt.load == 1:
        out_path = "/local/output/{}/best_start".format(opt.description)
        os.makedirs(out_path, exist_ok=True)
        val_report = validate_model(model_net, val_dataloader, save_disk=True, out_dir=out_path)

        wandb.log({
            "val_mse": val_report["MSE"],
            "val_psnr": val_report["PSNR"],
            "val_ssim": val_report["SSIM"],
        })

        best_psnr = val_report["PSNR"]
        print("Starting at - validation MSE: {:.3f} - PSNR: {:.3f} - SSIM: {:.4f}".format(val_report["MSE"],
                                                                                          val_report["PSNR"],
                                                                                          val_report["SSIM"]))
    else:
        os.makedirs(bst_ckp_path, exist_ok=True)

    round_loss = 0
    for i, batch in enumerate(train_dataloader):
        input_img = Variable(batch[0].type(Tensor))
        gt_img = Variable(batch[1].type(Tensor))

        optimizer.zero_grad()
        map = compute_maxchann_map(input_img, gt_img)
        out_img = model_net(input_img)
        loss = compute_loss(out_img, gt_img, opt, field_loss_module=field_loss_module)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_net.parameters(), 0.01)
        optimizer.step()

        round_loss += loss.detach().item()

        idx = i + startidx // opt.batch_size
        if (idx + 1) % opt.round_samples == 0:
            scheduler.step()

            round = (idx + 1) // opt.round_samples
            print("Round: {} - training_loss {:.4f}".format(round, round_loss))
            round_loss = 0

            if round % opt.valid_checkpoint == 0:
                save_disk = round % opt.save_checkpoint == 0

                if save_disk:
                    out_path = "/local/output/{}/{}".format(opt.description, round)
                    ckp_path = "/local/checkpoints/{}/{}".format(opt.description, round)

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
                if val_report["PSNR"] > best_psnr:
                    best_psnr = val_report["PSNR"]
                    save_checkpoint(bst_ckp_path, model_net, optimizer, scheduler)

                print("Round: {} - validation MSE: {:.3f} - PSNR: {:.3f} - SSIM: {:.4f}".format(round, val_report["MSE"],
                                                                                                val_report["PSNR"],
                                                                                                val_report["SSIM"]))




