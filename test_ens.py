import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image
from skimage.io import imsave
from spfnet import SPFNet
from utils import load_checkpoint, tensor_to_img

import torchvision.transforms.functional as  ft


def pad_if_needed(inp_image):
    c, h, w = inp_image.shape
    if h % 32 != 0 or w % 32 != 0:
        pad_dim = [0, 0, 0, 0]
        if w % 32 != 0:
            w32 = (w // 32 + 1) * 32
            pad_dim[1] = w32 - w

        if h % 32 != 0:
            h32 = (h // 32 + 1) * 32
            pad_dim[-1] = h32 - h

        print(pad_dim)
        return F.pad(inp_image, tuple(pad_dim), "reflect"), inp_image.shape
    else:
        return inp_image, None


def resize_if_needed(inp_image):
    shape_to = inp_image.shape

    c, w, h = inp_image.shape
    if h % 32 != 0 or w % 32 != 0:
        h32 = (h // 32) * 32
        w32 = (w // 32) * 32

        return F.interpolate(inp_image.unsqueeze(0), (w32, h32), mode='bicubic', align_corners=False)[0], shape_to
    else:
        return inp_image, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_pth", default="data/mipi24/test/inp", help="Input files dir")
    parser.add_argument("--ckp_pth", default="checkpoints/spfnet-direct-transfer-mipi24-alexssimloss/best/checkpoint.pt",
                        help="Model checkpoint path")
    parser.add_argument("--description", default="spfnet-direct-transfer-mipi24-alexssimloss-ensemble-tst")

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available():
        print("Using {} CUDA devices available".format(torch.cuda.device_count()))
        model_net = torch.nn.DataParallel(SPFNet(16, device="cuda"))
    else:
        model_net = torch.nn.DataParallel(SPFNet(16, device="cpu"))

    optimizer = torch.optim.Adam(model_net.parameters())
    model_net, _, _ = load_checkpoint(opt.ckp_pth, model_net, optimizer,
                                      torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[]))

    model_net.eval()

    files = [f for f in os.listdir(opt.in_pth) if f.endswith(".png")]
    to_tensor = torchvision.transforms.ToTensor()

    os.makedirs("submissions/{}".format(opt.description), exist_ok=True)

    with (torch.no_grad()):
        for f in files:
            # inp_img, shape_to = resize_if_needed(to_tensor(Image.open(os.path.join(opt.in_pth, f))))
            img_tensor = to_tensor(Image.open(os.path.join(opt.in_pth, f)))

            out_tens = torch.zeros_like(img_tensor).unsqueeze(0).cuda()

            for i in range(4):
                tensor_t = torch.rot90(img_tensor, i, [-2, -1])
                print(i, img_tensor.shape, tensor_t.shape)

                inp_img, shape_to = pad_if_needed(tensor_t)
                out_img = model_net(inp_img.unsqueeze(0))
                # out_img = torch.rot90(out_img, (4 - i) % 4, [-2, -1])

                if shape_to is not None:
                    _, th, tw = shape_to
                    # out_img = F.interpolate(out_img, (tw, th), mode="bicubic", align_corners=False)
                    out_tens += 0.25 * torch.rot90(out_img[:, :, :th, :tw], (4 - i) % 4, [-2, -1])
                else:
                    out_tens += 0.25 * torch.rot90(out_img, (4 - i) % 4, [-2, -1])

            out_img = tensor_to_img(out_tens.detach().squeeze(0))
            imsave("submissions/{}/{}".format(opt.description, f), out_img)
