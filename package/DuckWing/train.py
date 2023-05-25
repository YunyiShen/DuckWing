import torch
from os.path import join
from tqdm import tqdm
import torch.nn as nn

def train(model, optimizer, criterion, 
          train_loader, 
          val_loader, device, 
          num_epochs = 10, 
          save_path = "./"):
    best_val_acc = 0
    model.to(device)
    loss_list = []
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        for wing, head, belly, spp in tqdm(train_loader):
            #print(images.shape)
            wing, head, belly, spp = wing.to(device), \
                                       head.to(device), \
                                       belly.to(device), \
                                       spp.to(device)
            optimizer.zero_grad()
            outputs = model(wing, head, belly)
            loss = criterion(outputs, spp)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            total = 0
            correct = 0
            for wing, head, belly, spp  in tqdm(val_loader):
                wing, head, belly, spp = wing.to(device), \
                                       head.to(device), \
                                       belly.to(device), \
                                       spp.to(device)
                outputs = wing, head, belly, spp = wing.to(device), \
                                       head.to(device), \
                                       belly.to(device), \
                                       spp.to(device)
                _, predicted = torch.max(outputs.data, 1)
                total += spp.size(0)
                correct += (predicted == spp).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch: {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%")
        if accuracy > best_val_acc:
            torch.save(model.state_dict(), 
                       join(save_path, model.name + 
                            '_DuckWing.pth')
                       )
            best_val_acc = accuracy
    return loss_list
