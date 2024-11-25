# evaluate a smoothed classifier on a dataset
import sys
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import torch.nn as nn
import datetime
from resnet import ResNet18
from torch.autograd import Variable
import yaml
# from models import *
# Torchvision
import torchvision
import torchvision.transforms as transforms
from architectures import get_architecture
from torchvision.transforms import Resize
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--purification", type=bool, default=True, help="whether to use purification")
parser.add_argument("--train_classifier", type=bool, help="path to denoiser model config")
parser.add_argument("--methods", type=str, help="path to denoiser model config")
args = parser.parse_args()
print(args)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                            shuffle=True, num_workers=2)
def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class Adapated_Classifier(nn.Module):
    def __init__(self, classifier, adapter,num_classes):
        super(Adapated_Classifier, self).__init__()
        # self.dataset = dataset
        self.classifier = classifier
        
        self.adapter = adapter
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
      

    def forward(self, x):
       
        z_a = self.adapter(x)

        z = self.classifier(x)
       
        out = self.fc(z_a + z)
        return out



if __name__ == "__main__":
    for i in [1]:
        args.sigma = i


        # # test cohen smoothadv baseline
        if args.methods == "rs":
        #     classifer_checkpoint_path = "/home/jid20004/final_project/cohen_models/cifar10/resnet110/noise_{:.2f}/checkpoint.pth.tar".format(args.sigma)
            test_model_path = "/home/jid20004/final_project/PyTorch-CIFAR-10-autoencoder/logs/rs_{}/best_adapted_classifier.pkl".format(i)
        elif args.methods == "smoothadv":
        #     classifer_checkpoint_path = "/home/jid20004/final_project/smoothAdv_models/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_{:.2f}/checkpoint.pth.tar".format(args.sigma)
            test_model_path = "/home/jid20004/final_project/PyTorch-CIFAR-10-autoencoder/logs/smoothadv_{}/best_adapted_classifier.pkl".format(i)

        # classifer_checkpoint = torch.load(classifer_checkpoint_path)
        # base_classifier = get_architecture(classifer_checkpoint["arch"], "cifar10")
        # base_classifier.load_state_dict(classifer_checkpoint['state_dict'],strict=False)
        # base_classifier.eval()


        
        args.outfile = "/home/jid20004/final_project/PyTorch-CIFAR-10-autoencoder/certified_results/{}_denoise_{}_100.txt".format(args.methods,i)
    
        # test_model_path = "/home/jid20004/final_project/PyTorch-CIFAR-10-autoencoder/logs/smoothadv_{}/best_adapted_classifier.pkl".format(i)
        stat_dict = torch.load(test_model_path)
        adapated_classifer = Adapated_Classifier(get_architecture("cifar_resnet110", "cifar10"), ResNet18().cuda(), 10).cuda()
        adapated_classifer.load_state_dict(stat_dict['adapated_classifer'])



        smoothed_classifier = Smooth(adapated_classifer,get_num_classes(args.dataset), args.sigma)
    
        # prepare output file
        with torch.no_grad():
            f = open(args.outfile, 'w')
            print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

            # iterate through the dataset
            dataset = get_dataset(args.dataset, args.split)

            for i in range(len(testset)):

                # only certify every args.skip examples, and stop after args.max examples
                if i % args.skip != 0:
                    continue
                if i == args.max:
                    break

                (x, label) = testset[i]

                # (x1,label1) = testset[i]
                # import pdb; pdb.set_trace()
                
                before_time = time()
                # certify the prediction of g around x
                x = x.cuda()
                prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
                after_time = time()
                correct = int(prediction == label)

                time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
                print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                    i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

            f.close()
