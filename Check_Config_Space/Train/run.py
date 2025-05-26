import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from DatasetGenerator.FileIO import *
from DatasetGenerator.Generator import *
from Model import *
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch

from tqdm import tqdm
import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", default=-1)
# FLAGS = parser.parse_args()
local_rank = int(os.getenv('LOCAL_RANK', -1))

torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

device = torch.device("cuda", local_rank)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)        
    
if __name__ == "__main__":
    # Load the dataset
    train_data_dir = os.path.join(os.path.dirname(__file__), '../data/train')
    train_dataset = SceneDataset(train_data_dir)
    val_data_dir = os.path.join(os.path.dirname(__file__), '../data/val')
    val_dataset = SceneDataset(val_data_dir)
    test_data_dir = os.path.join(os.path.dirname(__file__), '../data/test')
    test_dataset = SceneDataset(test_data_dir)
    
    save_model_dir = os.path.join(os.path.dirname(__file__), '../model')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    # Create a DataLoader
    # train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=256, sampler=train_sampler, num_workers=8)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=256, sampler=val_sampler, num_workers=8)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, sampler=test_sampler, num_workers=8)
    
    # Initialize the model
    scene_feature_dim = 12  # Example dimension
    point_feature_dim = 4   # Example dimension
    model = CheckConfigSpaceModel(scene_feature_dim, point_feature_dim, attention_dim=64, num_heads=4, block_num=3).to(local_rank)
    
    # Initialize weights
    model.apply(init_weights)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # for _ in range(10):
        # # Train the model
        # train_model(model, train_loader, criterion, optimizer, num_epochs=1)
        
        # # Test the model
        # test_model(model, train_loader)
    model.train()
    num_epochs = 1000  # Number of epochs to train
    iterator = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in iterator:
        running_loss = 0.0
        train_loader.sampler.set_epoch(epoch)  # Set the epoch for the sampler
        for i, (scene_features, point_features, labels, valid_len) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # scene_features = scene_features.permute(1, 0, 2)
            point_features.unsqueeze_(1)
            # Forward pass
            outputs = model(scene_features.to(local_rank), point_features.to(local_rank), valid_len.to(local_rank))
            outputs.squeeze_(1)
            # outputs = torch.softmax(outputs, dim=1)
            # with torch.no_grad():
            #     check = torch.softmax(outputs, dim=1)
            #     check = check.cpu()
            #     print(check, labels)
            
            # Compute the loss
            loss = criterion(outputs, labels.to(local_rank))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if dist.get_rank() == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (scene_features, point_features, labels, mask) in enumerate(test_loader):
                    point_features.unsqueeze_(1)
                    outputs = model(scene_features.to(local_rank), point_features.to(local_rank), mask.to(local_rank)).squeeze_(1).cpu()
                    outputs = torch.softmax(outputs, dim=1)
                    total += labels.size(0)
                    predicted = torch.round(outputs)
                    res = [predicted[i, 0] == labels[i, 0] and predicted[i, 1] == labels[i, 1] for i in range(len(predicted))]
                    correct += sum(res)
            print(f"Test Accuracy: {100 * correct / total:.5f}%")
            
            ckp = model.module.state_dict()
            torch.save(ckp, os.path.join(save_model_dir, f"e{epoch+1}_acc{100 * correct / total:.5f}.pth"))
            
        
        