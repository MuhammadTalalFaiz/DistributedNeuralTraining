import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

def train(selected_model, selected_dataset):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    if selected_model == 'ResNet':
        model = torchvision.models.resnet50(pretrained=False)
    elif selected_model == 'DenseNet':
        model = torchvision.models.densenet121(pretrained=False)
    elif selected_model == 'VGG':
        model = torchvision.models.vgg16(pretrained=False)
    else:
        raise ValueError("Invalid model architecture selected")

    
    model.to(device)

    
    if selected_dataset == 'CIFAR-10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    else:
        raise ValueError("Invalid dataset selected")

    
    model = nn.parallel.DistributedDataParallel(model)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    
    model.train()
    training_progress = ""  
    for epoch in range(5):  
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}\n")

    
    checkpoint_path = f"{selected_model}_{selected_dataset}_checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)

    
    dist.destroy_process_group()

    return checkpoint_path, training_progress

def train_model(selected_model, selected_dataset):
   
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='gloo', init_method='tcp://localhost:12345', rank=0, world_size=1)
    
    
    return train(selected_model, selected_dataset)
