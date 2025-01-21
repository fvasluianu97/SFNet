import torch
import lpips
import os
from skimage.io import imread
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


if __name__ == '__main__':
    res_dir = "./results/flare7k++"
    imsize = (512, 512)
    solutions = [f for f in os.listdir(res_dir) if os.path.isdir(os.path.join(res_dir, f))]

    print("Found {} solutions:".format(len(solutions), solutions))

    loss_fn = lpips.LPIPS(net='alex', version='0.1')

    for s in solutions:
        print("Statistics for solution ", s)
        inp_dir = res_dir + "/" + s
        file_names = sorted([f for f in os.listdir(inp_dir) if f.endswith("_in.png")])
        num_samples = len(file_names)

        mse = 0
        xmse = 0
        psnr = 0
        xpsnr = 0

        ssim = 0
        xssim = 0

        olpips = 0
        xlpips = 0

        for f in file_names:
            f_id = f.split("_")[0]
            inp_img = imread("{}/{}_in.png".format(inp_dir, f_id))
            out_img = imread("{}/{}_out.png".format(inp_dir, f_id))
            gt_img = imread("{}/{}_gt.png".format(inp_dir, f_id))

            mse += mean_squared_error(out_img, gt_img)
            xmse += mean_squared_error(inp_img, gt_img)

            psnr += peak_signal_noise_ratio(gt_img, out_img)
            xpsnr += peak_signal_noise_ratio(gt_img, inp_img)

            ssim += structural_similarity(out_img, gt_img, channel_axis=-1)
            xssim += structural_similarity(inp_img, gt_img, channel_axis=-1)

            img0 = lpips.im2tensor(lpips.load_image("{}/{}_in.png".format(inp_dir, f_id)))
            img1 = lpips.im2tensor(lpips.load_image("{}/{}_out.png".format(inp_dir, f_id)))
            img2 = lpips.im2tensor(lpips.load_image("{}/{}_gt.png".format(inp_dir, f_id)))

            olpips += loss_fn.forward(img1, img2).item()
            xlpips += loss_fn.forward(img0, img2).item()


        mse /= num_samples
        xmse /= num_samples
        psnr /= num_samples
        xpsnr /= num_samples

        ssim /= num_samples
        xssim /= num_samples

        olpips /= num_samples
        xlpips /= num_samples


        print("MSE: {}|{} - PSNR {} | {} - SSIM {}|{} - LPIPS {}|{}".format(mse, xmse, psnr, xpsnr, ssim, xssim, olpips, xlpips))