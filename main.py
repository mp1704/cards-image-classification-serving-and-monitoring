import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import *
from dataset import *

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
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "gpu"
    print(f"Using {device}")
    print("-"*50)
    train_losses = []
    test_losses = []

    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for image, label in tqdm(train_dataloader):
            image = image.to(device)
            label = label.to(device)
            model = model.to(device)
            model.zero_grad()
            # forward pass
            prediction = model(image)
            loss = loss_fn(prediction, label)
            # backward
            running_loss += loss * image.shape[0] # loss * batch_size (it looks like AverageMeter())
            loss.backward()
            optimizer.step()
        train_loss = running_loss / len(train_dataloader) # loss of current epoch
        train_losses.append(train_loss)
        print(f"Epochs {epoch}: Training loss: {train_loss:.2f}")
        break