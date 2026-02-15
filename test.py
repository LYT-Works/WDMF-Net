import sys
sys.path.insert(0, '.')

import os
import time
import cv2
import numpy as np
from argparse import ArgumentParser
from PIL import Image

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter

from models.WDMFNet import BaseNet

# =========================================================
# Losses
# =========================================================
def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


def BCE(inputs, targets):
    return F.binary_cross_entropy(inputs, targets)


def Dice(inputs, targets):
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return 1 - dice


# =========================================================
# Validation (with inference time)
# =========================================================
@torch.no_grad()
def val(args, val_loader, model, epoch):
    model.eval()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    # ---------inference time stats---------
    infer_time_total = 0.0   # seconds
    infer_pairs = 0

    total_batches = len(val_loader)
    print('Total test batches:', total_batches)

    # ---------------warm-up----------------
    if args.onGPU and torch.cuda.is_available():
        warm_n = 20
        for it, (img, target) in enumerate(val_loader):
            pre_img = img[:, 0:3].cuda(non_blocking=True).float()
            post_img = img[:, 3:6].cuda(non_blocking=True).float()
            _ = model(pre_img, post_img)
            if it >= warm_n - 1:
                break
        torch.cuda.synchronize()

    # ---------------evaluation-------------
    for it, (img, target) in enumerate(val_loader):

        img_name = val_loader.sampler.data_source.file_list[it]
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU:
            pre_img = pre_img.cuda(non_blocking=True)
            post_img = post_img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        pre_img = pre_img.float()
        post_img = post_img.float()
        target = target.float()

        # ===============================
        #   Inference timing
        # ===============================
        if args.onGPU and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        output, output2, output3, output4 = model(pre_img, post_img)

        if args.onGPU and torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_time_total += (time.time() - t0)
        infer_pairs += pre_img.size(0)   # image pairs

        # -------- NOT counted below --------
        loss = (
            BCEDiceLoss(output, target) +
            BCEDiceLoss(output2, target) +
            BCEDiceLoss(output3, target) +
            BCEDiceLoss(output4, target)
        )
        epoch_loss.append(loss.item())

        pred = torch.where(output > 0.5,
                           torch.ones_like(output),
                           torch.zeros_like(output)).long()

        if args.onGPU and torch.cuda.device_count() > 1:
            pred = gather(pred, 0, dim=0)

        # save change maps
        pr = pred[0, 0].cpu().numpy()
        gt = target[0, 0].cpu().numpy()

        index_tp = np.where((pr == 1) & (gt == 1))
        index_fp = np.where((pr == 1) & (gt == 0))
        index_tn = np.where((pr == 0) & (gt == 0))
        index_fn = np.where((pr == 0) & (gt == 1))

        cmap = np.zeros([gt.shape[0], gt.shape[1], 3], dtype=np.uint8)
        cmap[index_tp] = [255, 255, 255]
        cmap[index_fp] = [255, 0, 0]
        cmap[index_tn] = [0, 0, 0]
        cmap[index_fn] = [0, 255, 0]

        Image.fromarray(cmap).save(args.vis_dir + img_name)

        f1 = salEvalVal.update_cm(pr, gt)

        if it % 5 == 0:
            avg_ms = (infer_time_total / max(1, infer_pairs)) * 1000.0
            print(
                '\r[%d/%d] F1: %.4f  loss: %.4f  infer: %.3f ms/pair'
                % (it, total_batches, f1, loss.item(), avg_ms),
                end=''
            )

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    avg_infer_ms = (infer_time_total / max(1, infer_pairs)) * 1000.0
    print("\nAverage Inference Time: %.3f ms / image pair" % avg_infer_ms)
    scores['infer_ms_per_pair'] = avg_infer_ms

    return average_epoch_loss_val, scores


# =========================================================
# Main validation pipeline
# =========================================================
def ValidateSegmentation(args):
    cudnn.benchmark = True
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)

    model = BaseNet()

    args.savedir = f"{args.savedir}_{args.file_root}_iter_{args.max_steps}_lr_{args.lr}/"
    args.vis_dir = f'./predict/{args.file_root}/'
    args.heatmap_dir = f'./heatmap/{args.file_root}/'

    if args.file_root in ['LEVIR', 'SYSU', 'WHU', 'LEVIR+', 'CLCD']:
        args.file_root = f'data/{args.file_root}'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs(args.heatmap_dir, exist_ok=True)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters:', total_params)

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset(
        "test", file_root=args.file_root, transform=valDataset
    )
    testLoader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False
    )

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'OA', 'IoU', 'F1', 'R', 'P'))
    logger.flush()


    model_file_name = args.savedir + 'best_model.pth'
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    loss_test, score_test = val(args, testLoader, model, 0)
    print("\nTest :\t OA (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test['OA'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('Test', score_test['OA'], score_test['IoU'],
                                                                   score_test['F1'], score_test['recall'],
                                                                   score_test['precision']))
    logger.flush()
    logger.close()

    torch.cuda.empty_cache()


# =========================================================
# Entry
# =========================================================
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR")
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--step_loss', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_mode', default='poly')
    parser.add_argument('--savedir', default='./results')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--logFile', default='testLog.txt')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--weight', default='', type=str)
    parser.add_argument('--ms', type=int, default=0)

    args = parser.parse_args()
    print('Args:', args)

    ValidateSegmentation(args)
