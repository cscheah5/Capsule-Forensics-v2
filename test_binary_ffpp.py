"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
sys.setrecursionlimit(15000)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import model_big
import time
import psutil

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databases/faceforensicspp', help='path to dataset')
parser.add_argument('--test_set', default ='test', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=21, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    vgg_ext = model_big.VggExtractor()
    capnet = model_big.CapsuleNet(2, opt.gpu_id)

    capnet.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.id) + '.pt')))
    capnet.eval()

    if opt.gpu_id >= 0:
        vgg_ext.cuda(opt.gpu_id)
        capnet.cuda(opt.gpu_id)


    ##################################################################################

    tol_label = np.array([], dtype=np.float64)
    tol_pred = np.array([], dtype=np.float64)
    tol_pred_prob = np.array([], dtype=np.float64)

    count = 0
    loss_test = 0

    # Initialize timing and resource variables
    total_inference_time = 0
    num_samples = 0

    # GPU memory tracking
    torch.cuda.reset_peak_memory_stats(opt.gpu_id)

    for img_data, labels_data in tqdm(dataloader_test):

        # Start timing
        start_time = time.time()

        labels_data[labels_data > 1] = 1
        img_label = labels_data.numpy().astype(np.float64)

        if opt.gpu_id >= 0:
            img_data = img_data.cuda(opt.gpu_id)
            labels_data = labels_data.cuda(opt.gpu_id)

        input_v = Variable(img_data)

        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=opt.random)

        # End timing
        end_time = time.time()
        batch_time = end_time - start_time
        total_inference_time += batch_time

        # Update total samples
        num_samples += img_data.size(0)

        output_dis = class_.data.cpu()
        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float64)

        for i in range(output_dis.shape[0]):
            if output_dis[i,1] >= output_dis[i,0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        
        pred_prob = torch.softmax(output_dis, dim=1)
        tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].data.numpy()))

        count += 1

    # Metrics
    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_test /= count
    precision = metrics.precision_score(tol_label, tol_pred)
    recall = metrics.recall_score(tol_label, tol_pred)
    f1 = metrics.f1_score(tol_label, tol_pred)
    confusion_matrix = metrics.confusion_matrix(tol_label, tol_pred)

    fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    max_memory_allocated = torch.cuda.max_memory_allocated(opt.gpu_id) / (1024**2)  # Convert to MB

    average_inference_time = total_inference_time / num_samples
    # fnr = 1 - tpr
    # hter = (fpr + fnr)/2

    print('[Epoch %d] Test acc: %.2f   EER: %.2f' % (opt.id, acc_test*100, eer*100))
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print(f"Confusion Matrix:\n{confusion_matrix}")
    print(f"Average Inference Time per Sample: {average_inference_time:.4f} seconds")
    print(f"Peak GPU Memory Usage: {max_memory_allocated:.2f} MB")
    text_writer.write('%d,%.2f,%.2f\n'% (opt.id, acc_test*100, eer*100))

    text_writer.flush()
    text_writer.close()
