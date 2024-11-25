# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
# Torchvision
import torchvision
import torchvision.transforms as transforms
from core import Smooth
# Matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
from time import time
import datetime
# OS
import os
import argparse
import sys
from architectures import get_architecture
from datasets import get_dataset, DATASETS, get_num_classes
from tqdm import tqdm
from resnet import resnet50,resnet18
sys.path.insert(0, '/home/jid20004/final_project/pytorch_resnet_cifar10')

import timm

torch.backends.cudnn.benchmark = True
class Adapated_Classifier(nn.Module):
    def __init__(self, classifier, adapter,num_classes):
        super(Adapated_Classifier, self).__init__()
        # self.dataset = dataset
        self.classifier = classifier
        self.adapter = adapter
        self.fc = nn.Linear(64, num_classes) ## 64 for original resnet

    def forward(self, x):
        z_a = self.adapter(x)
        z = self.classifier(x)
        out = self.fc(z_a+z)
        return out



def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Adapted Classifier Training")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--batch_size", default=256, type=int,help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--exp_name", default="test", type=str, help="Experiment name.")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode.")
    parser.add_argument("--noise_scale", default=0.50, type=float, help="Noise scale.")
    parser.add_argument("--methods", default="rs", type=str, help="Model name.")
    parser.add_argument("--certify", action="store_true", default=False, help="Certify mode.")
    parser.add_argument("--outfile", default="path", type=str, help="Output file.")
    args = parser.parse_args()
    
    log_file = open('logs/{}.txt'.format(args.exp_name), 'w')
    # if args.debug:
    #     sys.stdout = sys.__stdout__
    # else:
    #     sys.stdout = log_file
    print(args)
    
 
    # #tensorboad writer
    log_writer = SummaryWriter('logs/{}'.format(args.exp_name))     
    # # Load data
   
    # transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    # ])
    



    if args.methods == "vit":
        classifer_checkpoint_path = "timm_models/vit_base_patch16_224_in21k.pth"
        base_classifier = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes = 10)
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        # for param in base_classifier.parameters():
        #     param.requires_grad = False
    else:
        if args.methods == "rs":
            classifer_checkpoint_path = "/home/jid20004/final_project/cohen_models/cifar10/resnet110/noise_{:.2f}/checkpoint.pth.tar".format(args.noise_scale)
        elif args.methods == "smoothadv":
            classifer_checkpoint_path = "/home/jid20004/final_project/smoothAdv_models/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_{:.2f}/checkpoint.pth.tar".format(args.noise_scale)
        classifer_checkpoint = torch.load(classifer_checkpoint_path)
        base_classifier = get_architecture(classifer_checkpoint["arch"], "cifar10")
        base_classifier.load_state_dict(classifer_checkpoint['state_dict'])
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=4)


    adapter = resnet18().cuda()

    adapated_classifer = Adapated_Classifier(base_classifier, adapter, 10).cuda()
    adapated_classifer = base_classifier
    # Define an optimizer and criterion
    
    optimizer = optim.Adam(adapated_classifer.parameters())
    # optimizer = optim.AdamW(adapated_classifer.parameters(),lr=5e-05)
    # optimizer = optim.Adam([{'params':adapated_classifer.adapter.parameters(), 
    #                          'params':adapated_classifer.fc.parameters()}])
    # optimizer = optim.Adam(autoencoder.parameters())
    dropout = nn.Dropout(0)

    # Print all of the hyperparameters as variables of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print(args)
    print("Feature Extractor:", classifer_checkpoint_path)
    print("Optimizer:", optimizer)
    print(dropout)
    # using tensorboad to log the training process
    best_adapted_classifier = {"acc": 0}
    best_acc = 0

    for epoch in range(args.epochs):
        
        running_loss = 0
        # for i, (inputs, labels) in tqdm(enumerate(trainloader, 0)):
        #     loss = torch.tensor(0.0, requires_grad=True).cuda()
        #     inputs = get_torch_vars(inputs)
        #     labels = get_torch_vars(labels)
        #     sigma = args.noise_scale
        #     noise = torch.randn_like(inputs)*sigma
        #     noised_samples = (inputs + noise).cuda()
        #     noised_logits = adapated_classifer(noised_samples)
        #     noised_softmax = torch.nn.Softmax(dim=1)(noised_logits)
        #     loss = F.cross_entropy(noised_softmax, labels) 
        #     # ============ Forward ============
            
        #     # ============ Backward ============
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     # ============ Logging ============
        #     running_loss += loss.data
           
        # write the val code
        correct = 0
        total = 0
        test_loss = 0
        adapated_classifer.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = get_torch_vars(images)
                labels = get_torch_vars(labels)
                sigma = args.noise_scale
                noise = torch.randn_like(images)*sigma
                noised_samples = (images+noise)
                noised_logits = adapated_classifer(noised_samples)
                noised_softmax = torch.nn.Softmax(dim=1)(noised_logits)
                _, predicted = torch.max(noised_softmax.data, 1)
                test_loss += F.cross_entropy(noised_softmax, labels).item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            if correct/total > best_acc:
                best_acc = correct/total
                best_adapted_classifier['acc'] = best_acc
                best_adapted_classifier['adapated_classifer'] = adapated_classifer.state_dict()
                best_adapted_classifier['adapter'] = adapter.state_dict()
                # best_adapted_classifier['classifier'] = classifer_checkpoint['state_dict']
                best_adapted_classifier['epoch'] = epoch    
                best_adapted_classifier['optimizer'] = optimizer.state_dict()
                best_adapted_classifier['loss'] = test_loss/len(testloader)

                torch.save(best_adapted_classifier, "./logs/{}/best_adapted_classifier.pkl".format(args.exp_name))

        log_writer.add_scalar('training loss', running_loss / len(trainloader), epoch)
        log_writer.add_scalar('test loss', test_loss / len(testloader), epoch)
        log_writer.add_scalar('test accuracy', 100 * correct / total, epoch)
        print("Epoch %d" % epoch, 'Train Loss %.3f' % ( running_loss / len(trainloader)), 'Test Loss %.3f' % (test_loss/len(testloader)), 'Test Accuracy: %f %%' % (100 * correct / total))
   
    log_writer.close()
    print('Finished Training')

    log_file.close()

if __name__ == '__main__':
    
    main()
