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
    capnet = model_big.CapsuleNet(2, opt.gpu_id) # Assuming 2 classes (real/fake)

    capnet.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.id) + '.pt')))
    capnet.eval() # Set model to evaluation mode

    if opt.gpu_id >= 0:
        vgg_ext.cuda(opt.gpu_id)
        capnet.cuda(opt.gpu_id)

    ##################################################################################

    # Store labels and predictions for all metrics
    all_labels = []
    all_outputs = [] # To store raw model outputs (logits/scores) for top-k

    # Initialize timing and resource variables
    total_inference_time = 0
    num_samples = 0

    # GPU memory tracking
    if opt.gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(opt.gpu_id)

    with torch.no_grad(): # Disable gradient calculation for inference
        for img_data, labels_data in tqdm(dataloader_test):

            # Start timing
            start_time = time.time()

            # Ensure labels are 0 or 1 for binary classification
            labels_data[labels_data > 1] = 1 
            
            if opt.gpu_id >= 0:
                img_data = img_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            input_v = Variable(img_data)

            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=opt.random) # classes are activations, class_ are final scores/logits

            # End timing
            end_time = time.time()
            batch_time = end_time - start_time
            total_inference_time += batch_time

            # Update total samples
            num_samples += img_data.size(0)

            # Collect labels and model outputs for overall metrics
            all_labels.append(labels_data.cpu())
            all_outputs.append(class_.data.cpu()) # Keep as tensor for top-k

    # Concatenate all collected data
    tol_label = torch.cat(all_labels).numpy().astype(np.float64)
    all_outputs_tensor = torch.cat(all_outputs)

    # --- Calculate metrics ---

    # Get predicted classes (0 or 1) for accuracy, precision, recall, F1, confusion matrix
    # This is equivalent to your original 'output_pred' logic
    _, predicted_classes = torch.max(all_outputs_tensor, 1)
    tol_pred = predicted_classes.numpy().astype(np.float64)

    # Get probabilities for EER and ROC curve (assuming class 1 is the positive class)
    tol_pred_prob = torch.softmax(all_outputs_tensor, dim=1)[:, 1].numpy()

    # Standard metrics
    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    precision = metrics.precision_score(tol_label, tol_pred)
    recall = metrics.recall_score(tol_label, tol_pred)
    f1 = metrics.f1_score(tol_label, tol_pred)
    confusion_matrix = metrics.confusion_matrix(tol_label, tol_pred)

    fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # --- Top-1 and Top-5 Accuracy Calculation ---
    # Convert true labels to long for topk comparison
    true_labels_long = torch.from_numpy(tol_label).long() 
    
    # Calculate top-k accuracies
    # k can be 1, 5, etc.
    # Note: For 2 classes, top-1 is standard accuracy. Top-5 will be same as Top-1
    # unless you have more than 5 classes total in your model_big.CapsuleNet definition.
    # Assuming CapsuleNet(2, ...) means it outputs 2 scores/logits.
    
    # Get the top-k predicted class indices
    # We only have 2 classes, so k should be at most 2.
    # If CapsuleNet was defined with more than 2 classes, this would be more relevant.
    k = min(5, all_outputs_tensor.shape[1]) # Cap k at the actual number of classes

    _, topk_preds = all_outputs_tensor.topk(k, 1, True, True) # values, indices
    
    # Expand true_labels_long to match dimensions of topk_preds for comparison
    true_labels_expanded = true_labels_long.view(-1, 1).expand_as(topk_preds)
    
    # Check if true label is in top-k predictions
    correct_topk = (true_labels_expanded == topk_preds).any(dim=1)
    
    top1_accuracy = (topk_preds[:, 0] == true_labels_long).sum().item() / len(true_labels_long)
    topk_accuracy = correct_topk.sum().item() / len(true_labels_long)

    # --- Resource Usage ---
    max_memory_allocated = 0
    if opt.gpu_id >= 0 and torch.cuda.is_available():
        max_memory_allocated = torch.cuda.max_memory_allocated(opt.gpu_id) / (1024**2) 

    average_inference_time = total_inference_time / num_samples if num_samples > 0 else 0

    # --- Print Results ---
    print("\n--- Evaluation Results ---")
    print(f"[Checkpoint ID {opt.id}]")
    print(f"Test Accuracy: {acc_test*100:.2f}%")
    print(f"Top-1 Accuracy: {top1_accuracy*100:.2f}%")
    # Only print Top-K if K is greater than 1 (and relevant for the number of classes)
    if k > 1 and k <= all_outputs_tensor.shape[1]:
        print(f"Top-{k} Accuracy: {topk_accuracy*100:.2f}%") 
    else:
        # If k is 1 or if model only has 2 classes, Top-K is effectively Top-1
        print(f"Top-5 Accuracy is equivalent to Top-1 Accuracy for a 2-class problem.")

    print(f"EER: {eer*100:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix}")
    print(f"Average Inference Time per Sample: {average_inference_time:.4f} seconds")
    print(f"Peak GPU Memory Usage: {max_memory_allocated:.2f} MB")
    
    text_writer.write(f"{opt.id},{acc_test*100:.2f},{eer*100:.2f},{top1_accuracy*100:.2f},{topk_accuracy*100:.2f}\n")

    text_writer.flush()
    text_writer.close()