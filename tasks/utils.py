import os
import torch
from env.base import device

work_dir = os.path.join(os.path.dirname(__file__), "../runs")


def create_no_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_workspace(task_name):
    create_no_exists(work_dir)
    workspace = os.path.join(work_dir, task_name)
    create_no_exists(workspace)
    log_dir = os.path.join(workspace, "logs")
    create_no_exists(log_dir)
    models_dir = os.path.join(workspace, "model_files")
    create_no_exists(models_dir)
    return task_name, workspace, log_dir, models_dir


def cal_ssim(im1, im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = torch.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = torch.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def ssim(t1, t2):
    """
    :param t1: n,w,h
    :param t2: n,w,h
    :return: cal_ssim / n
    """
    result = torch.zeros(1).to(device)
    for i in range(len(t1)):
        result += cal_ssim(t1[i], t2[i])
    return result / len(t1)
