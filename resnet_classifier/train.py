import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import datasets, transforms, models  
from torch.utils.data import DataLoader  
from sklearn.metrics import accuracy_score, classification_report  
import os  
import copy  
  
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print("Device: ", device)  

   
transform = transforms.Compose([  
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])  
  
 
train_dataset = datasets.ImageFolder(root='MEFLUT-resnet/train', transform=transform)  
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  
  
val_dataset = datasets.ImageFolder(root='MEFLUT-resnet/val', transform=transform)  
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)  
  
 
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features  
model.fc = nn.Linear(num_ftrs, 2)  # dark:1;  bright:0 
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)  


num_epochs = 10  
best_val_acc = 0.0 
best_model_wts = copy.deepcopy(model.state_dict()) 
  
for epoch in range(num_epochs):  
    model.train()  
    running_loss = 0.0  
    correct_preds = 0  
    total_preds = 0  
  
    for inputs, labels in train_loader:  
        inputs, labels = inputs.to(device), labels.to(device)  
  
        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
  
        running_loss += loss.item() * inputs.size(0)  
        _, preds = torch.max(outputs, 1)  
        correct_preds += torch.sum(preds == labels.data)  
        total_preds += labels.size(0)  
  
    epoch_loss = running_loss / len(train_loader.dataset)  
    epoch_acc = correct_preds.double() / total_preds  
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')  
  
    # 评估模型  
    model.eval()  
    val_correct_preds = 0  
    val_total_preds = 0  
    with torch.no_grad():  
        for inputs, labels in val_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)  
            _, preds = torch.max(outputs, 1)  
              
            val_correct_preds += torch.sum(preds == labels.data)  
            val_total_preds += labels.size(0)  
  
    val_acc = val_correct_preds.double() / val_total_preds  
    print(f'Validation Accuracy: {val_acc:.4f}')  
  
    if val_acc > best_val_acc:  
        best_val_acc = val_acc  
        best_model_wts = copy.deepcopy(model.state_dict())  
  
  
model.load_state_dict(best_model_wts)  
best_model_path = 'classifier.pth'  
torch.save(model.state_dict(), best_model_path)  
print(f'Best model saved to {best_model_path}')  