import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import *
from dataset import *
from utils import *
if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128))
    ])
    train_folder = "./cards-image-datasetclassification/train"
    test_folder = "./cards-image-datasetclassification/test"

    train_dataset = PlayingCardDataset(data_dir = train_folder,
                                    transform = transform)
    test_dataset = PlayingCardDataset(data_dir = test_folder,
                                    transform = transform)

    train_dataloader = DataLoader(dataset = train_dataset,
                                batch_size = 32,
                                shuffle = True)
    test_dataloader = DataLoader(dataset = test_dataset,
                                batch_size = 32,
                                shuffle = False)
    
    model = MyModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "gpu"
    model = model.to(device)
    print(f"Using {device}")
    print("-"*50)
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        running_loss = 0.
        train_acc = 0.
        model.train()   
        for image, label in tqdm(train_dataloader, desc = "training .."):
            image = image.to(device)
            label = label.to(device)
            model.zero_grad()
            # forward pass
            prediction = model(image)
            loss = loss_fn(prediction, label)
            train_acc += accuracy_fn(y_true=label,
                                y_pred = prediction.argmax(dim=-1))
            # backward
            running_loss += loss * image.shape[0] # loss * batch_size (it looks like AverageMeter())
            loss.backward()
            optimizer.step()
            # break # FOR QUICK TEST
        train_loss = running_loss / len(train_dataloader) # loss of current epoch
        train_losses.append(train_loss)
        train_acc /= len(train_dataloader)

        
        model.eval()
        test_running_loss = 0.
        test_acc = 0.
        with torch.inference_mode():
            for image, label in tqdm(test_dataloader, desc = "testing ..."):
                image = image.to(device)
                label = label.to(device)
                prediction = model(image)
                loss = loss_fn(prediction, label)
                test_running_loss += loss * image.shape[0]
                test_acc += accuracy_fn(y_true=label,
                                y_pred = prediction.argmax(dim=-1))
        test_loss = test_running_loss / len(test_dataloader)
        test_losses.append(test_loss)
        test_acc /= len(test_dataloader)
        print(f"Epochs {epoch+1}: Training loss: {train_loss:.2f} || Test loss: {test_loss:2f}")
        print(f"\t Train acc: {train_acc:2f} || Test acc: {test_acc:2f}")

