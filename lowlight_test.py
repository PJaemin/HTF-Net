import torch
import torchvision
import torch.optim
import os
from Model import Image_network
import numpy as np
from PIL import Image
import glob
import time
from matplotlib import pyplot as plt
import cv2
from unet_model import UNet
import torch.nn.functional as F
from IQA_pytorch import SSIM, MS_SSIM, FSIM, LPIPSvgg
from Metrics import cal_PSNR,PSNR1,findFile


gt_path = './data/test_gt/'
ssim_model = SSIM(channels=3).cuda()
msssim_model = MS_SSIM(channels=3).cuda()
fsim_model = FSIM(channels=3).cuda()
lpips_model = LPIPSvgg().cuda()

def get_hist(file_name):
    src = cv2.imread(file_name)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    hist_s = np.zeros((3, 256))

    for (j, color) in enumerate(("red", "green", "blue")):
        S = src[..., j]
        hist_s[j, ...], _ = np.histogram(S.flatten(), 256, [0, 256])
        hist_s[j, ...] = hist_s[j, ...] / np.sum(hist_s[j, ...])

    hist_s = torch.from_numpy(hist_s).float()

    return hist_s


def lowlight(image_path):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 0번 GPU에 메모리 할당
    data_lowlight = Image.open(image)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight * 2.0 - 1.0
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    attnet = UNet(3, 3)
    attnet = attnet.cuda()
    attnet.eval()
    attnet.load_state_dict(torch.load('models/att_final.pth', map_location='cuda:0'))

    Imgnet = Image_network()
    Imgnet = Imgnet.cuda()
    Imgnet.eval()
    Imgnet.load_state_dict(torch.load('models/Img_final.pth', map_location='cuda:0'))

    hist = get_hist(image)
    hist = hist.cuda().unsqueeze(0)
    hist = hist * 2.0 - 1.0

    start = time.time()
    att = attnet(F.interpolate(data_lowlight, 256))
    att = att.cuda()
    enhanced_img, vec = Imgnet(data_lowlight,att,hist)

    end_time = (time.time() - start)

    result_path = image_path.replace('test_data', 'result')

    plot_path = image_path.replace('test_data', 'test_plots')

    if not os.path.exists(result_path.replace('/' + result_path.split("/")[-1], '')):
        os.makedirs(result_path.replace('/' + result_path.split("/")[-1], ''))
    (fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    # print(vec.shape)
    vec = vec.squeeze(0)
    # print(vec.shape)
    vec = vec * 0.5 + 0.5
    vec = vec.cpu()
    vec = vec.numpy()
    vec = vec * 255
    r = vec[0, ...]
    g = vec[1, ...]
    b = vec[2, ...]
    axs[0].plot(r, color='r',linewidth=4)
    axs[1].plot(g, color='g',linewidth=4)
    axs[2].plot(b, color='b',linewidth=4)
    axs[0].set_xlim([0, 255])
    axs[0].set_ylim([0, 255])
    axs[1].set_xlim([0, 255])
    axs[1].set_ylim([0, 255])
    axs[2].set_xlim([0, 255])
    axs[2].set_ylim([0, 255])
    plt.tight_layout()
    plt.draw()
    plt.savefig(plot_path+'.pdf')



    gt_file = gt_path + str(findFile(gt_path, os.path.basename(image)))
    data_gt = Image.open(gt_file)
    data_gt = (np.asarray(data_gt) / 255.0)
    data_gt = data_gt * 2.0 - 1.0
    data_gt = torch.from_numpy(data_gt).float()
    # data_lowlight = data_lowlight * 2.0 - 1.0
    data_gt = data_gt.permute(2, 0, 1)
    data_gt = data_gt.cuda().unsqueeze(0)
    psnr = PSNR1(enhanced_img, data_gt)
    ssimt = ssim_model(enhanced_img, data_gt, as_loss=False)
    msssimt = msssim_model(enhanced_img, data_gt, as_loss=False)
    fsimt = fsim_model(enhanced_img, data_gt, as_loss=False)
    lpipst = lpips_model(enhanced_img, data_gt, as_loss=False)
    ssim = ssimt.item()
    msssim = msssimt.item()
    fsim = fsimt.item()
    lpips = lpipst.item()

    enhanced_image = enhanced_img * 0.5 + 0.5
    torchvision.utils.save_image(enhanced_image, result_path)

    return psnr, ssim, msssim, fsim, lpips





# 파일이 import에 의해서가 아닌 interpreter에 의해서 호출될때만 실행 가능
if __name__ == '__main__':
    # test_images
    with torch.no_grad():  # gradient 연산 옵션을 끔. 이 내부 컨텍스트 텐서들은 requires_grad=False 되어 메모리사용 아낌
        filePath = 'data/test_data/'  # test dataset path

        file_list = os.listdir(filePath)  # os.listdir은 디렉토리내에 모든 파일과 디렉토리 리스트를 리턴함

        best = 0
        ep = 0
        sum_psnr = 0
        sum_ssim = 0
        sum_msssim = 0
        sum_fsim = 0
        sum_lpips = 0
        for file_name in file_list:  # DCIM,LIME까지
            test_list = glob.glob(filePath + file_name + "/*")  # filePath+file_name에 해당되는 모든 파일
            n_of_files = len(test_list)
            for image in test_list:
                # image = image
                # print(image)
                psnr, ssim, msssim, fsim, lpips = lowlight(image)
                sum_psnr += psnr
                sum_ssim += ssim
                sum_msssim += msssim
                sum_fsim += fsim
                sum_lpips += lpips
                print('[Done] ' + str(image))
                # result_path = image.replace('test_data', 'result')
                # img_psnr = cal_PSNR(result_path)
                # print(str(image) + '\tPSNR: '+str(img_psnr))
                # sum_psnr += img_psnr
            avg_psnr = sum_psnr / n_of_files
            avg_ssim = sum_ssim / n_of_files
            avg_msssim = sum_msssim / n_of_files
            avg_fsim = sum_fsim / n_of_files
            avg_lpips = sum_lpips / n_of_files

            print("PSNR:\t" + str(avg_psnr))
            print("SSIM:\t" + str(avg_ssim))
            print("MSSSIM:\t" + str(avg_msssim))
            print("FSIM:\t" + str(avg_fsim))
            print("LPIPS:\t" + str(avg_lpips))
