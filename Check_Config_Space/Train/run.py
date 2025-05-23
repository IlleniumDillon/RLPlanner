import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from DatasetGenerator.FileIO import *
from DatasetGenerator.Generator import *
from Model import *

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)

def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (scene_features, point_features, labels, valid_len) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # scene_features = scene_features.permute(1, 0, 2)
            point_features.unsqueeze_(1)
            # Forward pass
            outputs = model(scene_features.to(device), point_features.to(device), valid_len.to(device))
            outputs.squeeze_(1)
            # outputs = torch.softmax(outputs, dim=1)
            
            # Compute the loss
            loss = criterion(outputs, labels.to(device))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # print(f"Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (scene_features, point_features, labels, mask) in enumerate(test_loader):
            point_features.unsqueeze_(1)
            outputs = model(scene_features.to(device), point_features.to(device), mask.to(device)).squeeze_(1).cpu()
            outputs = torch.softmax(outputs, dim=1)
            total += labels.size(0)
            predicted = torch.round(outputs)
            res = [predicted[i, 0] == labels[i, 0] and predicted[i, 1] == labels[i, 1] for i in range(len(predicted))]
            correct += sum(res)
    print(f"Test Accuracy: {100 * correct / total:.5f}%")
        
    
if __name__ == "__main__":
    # Load the dataset
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    dataset = SceneDataset(data_dir)
    
    # Create a DataLoader
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize the model
    scene_feature_dim = 12  # Example dimension
    point_feature_dim = 4   # Example dimension
    model = CheckConfigSpaceModel(scene_feature_dim, point_feature_dim, attention_dim=64, num_heads=2, block_num=2).to(device)
    
    # Initialize weights
    model.apply(init_weights)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # for _ in range(10):
        # # Train the model
        # train_model(model, train_loader, criterion, optimizer, num_epochs=1)
        
        # # Test the model
        # test_model(model, train_loader)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (scene_features, point_features, labels, valid_len) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # scene_features = scene_features.permute(1, 0, 2)
            point_features.unsqueeze_(1)
            # Forward pass
            outputs = model(scene_features.to(device), point_features.to(device), valid_len.to(device))
            outputs.squeeze_(1)
            # outputs = torch.softmax(outputs, dim=1)
            # with torch.no_grad():
            #     check = torch.softmax(outputs, dim=1)
            #     check = check.cpu()
            #     print(check, labels)
            
            # Compute the loss
            loss = criterion(outputs, labels.to(device))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # print(f"Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (scene_features, point_features, labels, mask) in enumerate(train_loader):
                point_features.unsqueeze_(1)
                outputs = model(scene_features.to(device), point_features.to(device), mask.to(device)).squeeze_(1).cpu()
                outputs = torch.softmax(outputs, dim=1)
                total += labels.size(0)
                predicted = torch.round(outputs)
                res = [predicted[i, 0] == labels[i, 0] and predicted[i, 1] == labels[i, 1] for i in range(len(predicted))]
                correct += sum(res)
        print(f"Test Accuracy: {100 * correct / total:.5f}%")
        
        