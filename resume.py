import argparse
import os
import torch
import torchvision.transforms as transforms
from dataloader import ImageSet
from loss import compute_loss
from model import DSMNet, Image2ImageTranslator
from perceptual_loss import PerceptualLossModule
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils import validate_model, save_checkpoint, load_checkpoint
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_epoch", type=int, default=300, help="Epoch to resume training from")
    parser.add_argument("--end_epoch", type=int, default=400, help="Epoch to end training")
    parser.add_argument("--decay_epoch", type=int, default=300, help="epoch from which to start lr decay")
    parser.add_argument("--n_steps", type=int, default=2, help="number of step decays for the learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")

    parser.add_argument("--alpha_1", type=float, default=0.5, help="weight for SSIM loss")
    parser.add_argument("--alpha_2", type=float, default=0.6, help="weight for gradient loss")
    parser.add_argument("--alpha_3", type=float, default=0.2, help="weight for weights map loss")
    parser.add_argument("--alpha_4", type=float, default=0.75, help="weight for content loss")
    parser.add_argument("--beta_1", type=float, default=2.0, help="effect estimator loss multiplier")

    parser.add_argument("--img_height", type=int, default=320, help="size of image height")
    parser.add_argument("--img_width", type=int, default=480, help="size of image width")
    parser.add_argument("--valid_checkpoint", type=int, default=5, help="checkpoint for visual inspection")
    parser.add_argument("--save_checkpoint", type=int, default=10, help="Save checkpoint interval")
    parser.add_argument("--description", default="gradual-model-spatial-l1", help="Experiment name")


    opt = parser.parse_args()

    # TODO customize block_size through params
    block_sz = [32, 64, 128, 256, 128, 64, 32]
    model_net = Image2ImageTranslator(block_sz)

    # TODO implement equivalent cropping to images instead resizing
    transforms_train = [
        transforms.Resize((opt.img_height, opt.img_width), Image.Resampling.BICUBIC),
        transforms.ToTensor(),
    ]

    transforms_val = [
        transforms.Resize((opt.img_height, opt.img_width), Image.Resampling.BICUBIC),
        transforms.ToTensor(),
    ]

    train_dataloader = DataLoader(
        ImageSet("data/flare-removal", "train", transforms_=transforms_train), batch_size=opt.batch_size,
        shuffle=True, num_workers=opt.n_cpu)

    val_dataloader = DataLoader(
        ImageSet("data/flare-removal", "validation", transforms_=transforms_val), batch_size=1,
        shuffle=False, num_workers=opt.n_cpu)

    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
        model_net = model_net.cuda()
        field_loss_module = PerceptualLossModule(device=torch.device("cuda"))
    else:
        Tensor = torch.FloatTensor
        field_loss_module = PerceptualLossModule(device=torch.device("cpu"))

    betas = (opt.b1, opt.b2)
    optimizer = torch.optim.Adam(model_net.parameters(), lr=opt.lr, betas=betas)

    step_sz = (opt.end_epoch - opt.decay_epoch) // opt.n_steps
    milestones = [k * step_sz for k in range(opt.n_steps)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.6)

    load_model_checkpoint = "checkpoints/{}/best/checkpoint.pt".format(opt.description)
    model_net, optimizer, scheduler = load_checkpoint(load_model_checkpoint, model_net, optimizer, scheduler)


    val_report = validate_model(model_net, val_dataloader, save_disk=False, out_dir=None)
    best_psnr = val_report["PSNR"]
    print("Starting at - validation MSE: {:.3f} - PSNR: {:.3f} - SSIM: {:.4f}".format(val_report["MSE"],
                                                                                    val_report["PSNR"],
                                                                                    val_report["SSIM"]))

    bst_ckp_path = "checkpoints/{}/best".format(opt.description)

    for epoch in range(opt.start_epoch, opt.end_epoch):
        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            input_img = Variable(batch[0].type(Tensor))
            gt_img = Variable(batch[1].type(Tensor))

            optimizer.zero_grad()
            map, out_img = model_net(input_img)
            # loss = compute_loss(input_img, out_img, map, gt_img, opt, field_loss_module=field_loss_module)
            loss = torch.nn.functional.l1_loss(out_img, gt_img)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        if epoch > opt.decay_epoch:
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

            if val_report["PSNR"] > best_psnr:
                best_psnr = val_report["PSNR"]
                save_checkpoint(bst_ckp_path, model_net, optimizer, scheduler)

            print(
                "Epoch: {} - validation MSE: {:.3f} - PSNR: {:.3f} - SSIM: {:.4f}".format(epoch + 1, val_report["MSE"],
                                                                                          val_report["PSNR"],
                                                                                          val_report["SSIM"]))
