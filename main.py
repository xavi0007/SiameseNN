# import the necessary libraries
import numpy as np
import torch
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils
import config
from utilis import imshow, show_plot, SiameseDataset
from sklearn.model_selection import train_test_split
from contrastiveloss import ContrastiveLoss
import torchvision
from model import SiameseNetwork

# load the dataset
model_save_dir = config.model_dir
training_dir = config.training_dir
testing_dir = config.testing_dir
training_csv = config.training_csv
testing_csv = config.testing_csv
device_str = config.device

if device_str == 'mps' and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("using mps")
elif device_str == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def train_val_dataset(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

# Viewing the sample of images and to check whether its loading properly
def visualise(siamese_dataset):
    vis_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())


#train function
def train(train_dataloader,model,criterion,optimizer):
    loss=[] 
    model.train()
    model.to(device)
    for _, data in enumerate(train_dataloader,0):
      img0, img1 , label = data
      img0, img1 , label = img0.to(device), img1.to(device), label.to(device)
      optimizer.zero_grad()
      output1,output2 = model(img0,img1)
      loss_contrastive = criterion(output1,output2,label)
      loss_contrastive.backward()
      optimizer.step()
      loss.append(loss_contrastive.item())
    loss = np.array(loss)
    return loss.mean()/len(train_dataloader)


def eval(eval_dataloader,model,criterion):
    model.eval()
    model.to(device)
    if device_str == 'cuda':
        torch.cuda.empty_cache()
    loss=[] 
    counter=[]
    iteration_number = 0
    for i, data in enumerate(eval_dataloader,0):
      img0, img1 , label = data
      img0, img1 , label = img0.to(device), img1.to(device), label.to(device)
      output1,output2 = model(img0,img1)
      loss_contrastive = criterion(output1,output2,label)
      loss.append(loss_contrastive.item())
    loss = np.array(loss)
    return loss.mean()/len(eval_dataloader)

def model_test(test_dataloader):
    # Load the test dataset
    count = 0
    for i, data in enumerate(test_dataloader, 0):
        x0, x1, label = data
        concat = torch.cat((x0, x1), 0)
        output1, output2 = model(x0.to(device), x1.to(device))

        eucledian_distance = F.pairwise_distance(output1, output2)

        if label == torch.FloatTensor([[0]]):
            label = "Original Pair Of Signature"
        else:
            label = "Forged Pair Of Signature"

        imshow(torchvision.utils.make_grid(concat))
        print("Predicted Eucledian Distance:-", eucledian_distance.item())
        print("Actual Label:-", label)
        count = count + 1
        if count == 10:
            break

if __name__ == '__main__':
   # Load the the dataset from raw image folders
    siamese_dataset = SiameseDataset(
        training_csv,
        training_dir,
        transform=transforms.Compose(
            [transforms.Resize((105, 105)), transforms.ToTensor()]
        ),
    )
    #split train set into train and evaluate
    siamese_dataset = train_val_dataset(siamese_dataset)
    print(len(siamese_dataset['train']))
    print(len(siamese_dataset['val']))

     # Load the dataset as pytorch tensors using dataloader
    train_dataloader = DataLoader(siamese_dataset['train'],
                            shuffle=True,
                            num_workers=max(os.cpu_count(),4),
                            batch_size=config.batch_size) 
    eval_dataloader = DataLoader(siamese_dataset['val'],
                            shuffle=True,
                            num_workers=max(os.cpu_count(),4),
                            batch_size=config.batch_size) 
    
    model = SiameseNetwork().to(device, non_blocking=True)
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.max_lr, weight_decay=0.0005)
    
    for epoch in range(1,config.epochs):
        best_eval_loss = 99999
        train_loss = train(train_dataloader,model,criterion,optimizer)
        #evaluate every x rounds
        if epoch % 8 == 0:
            eval_loss = eval(eval_dataloader,model,criterion)
            print(f"Eval loss{eval_loss}")
            if eval_loss<best_eval_loss:
                best_eval_loss = eval_loss
                print("-"*20)
                print(f"Best Eval loss{best_eval_loss}")
                model_path =  model_save_dir+"/content/model.pth"
                torch.save(model.state_dict(), model_path)
                print("Model Saved Successfully") 
                print("-"*20)

        print(f"Training loss {train_loss}")
        print("-"*20)
   
    model_path =  model_save_dir+"/content/model.pth"
    torch.save(model.state_dict(), model_path)
    print("Model Saved Successfully") 
    
    test_dataset = SiameseDataset(
            training_csv=testing_csv,
            training_dir=testing_dir,
            transform=transforms.Compose(
                [transforms.Resize((105, 105)), transforms.ToTensor()]
            ),
        )

    test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)
    model_test(test_dataloader)