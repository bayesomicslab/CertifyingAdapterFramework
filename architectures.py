import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from datasets import get_normalize_layer
from torch.nn.functional import interpolate

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)


import time
import torch
import torch.nn as nn
import torch.optim as optim

def simulate_training(model, num_epochs=5, batch_size=128, input_shape=(3, 32, 32), num_classes=10, learning_rate=0.1):
    """
    Simulate training of a model using fake data and calculate average runtime over epochs.

    :param model: The PyTorch model to train.
    :param num_epochs: Number of epochs to train.
    :param batch_size: Batch size for training.
    :param input_shape: Shape of the input data.
    :param num_classes: Number of classes for classification.
    :param learning_rate: Learning rate for the optimizer.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Training loop
    model.train()
    epoch_times = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        for i in range(100):  # Simulate 100 batches per epoch
            # Generate fake data
            inputs = torch.randn(batch_size, *input_shape).cuda()
            labels = torch.randint(0, num_classes, (batch_size,)).cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds')

    average_time = sum(epoch_times) / num_epochs
    print(f'Average epoch time over {num_epochs} epochs: {average_time:.2f} seconds')

if __name__ == "__main__":
    # get resnet 18 from torch vision
    from torchvision.models import resnet18, resnet50, resnet101
    model = resnet101(pretrained=False).cuda()
    simulate_training(model)
    # simulate_training(get_architecture("cifar_resnet110", "cifar10"))