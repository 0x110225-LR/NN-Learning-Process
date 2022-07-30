# Simulation script by LR
# Only used to learning
import torchvision
from torch.utils.tensorboard import SummaryWriter
import Net
import torch
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root = "dataset", train = True, transform = torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root = "dataset", train = False, transform = torchvision.transforms.ToTensor(),
                                          download=True)

# Get length
train_num = len(train_data)
test_num = len(test_data)
print("train_set got {} data".format(train_num))
print("test_set got {} data".format(test_num))

# Use the DataLoader to load the dataset
train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

# Create instance
Lr = Net.Lr()

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
# lr(learning_rate) = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(Lr.parameters(), lr = learning_rate)

# Setting parameters in Network
# Recording train steps
train_step = 0
# Recording test steps
test_step = 0
# Epoch number
epoch = 10

# Recording in tensorboard
writer = SummaryWriter("common_nn")

for i in range(epoch):
    print("Starting No.{} epoch training...".format(i+1))
    
    # Train start
    Lr.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = Lr(imgs)
        loss = loss_fn(outputs, targets)

        # Optimizing
        optimizer.zero_grad()  # Very important!
        loss.backward()  # Very important!
        optimizer.step()  # Very important!

        train_step = train_step + 1
        if train_step % 100 == 0:
            print("Trained {} times, Loss:{}".format(train_step, loss.item()))
            writer.add_scalar("train_loss",loss.item(), train_step)
    
    # Test start
    Lr.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # Very important!
        for data in test_dataloader:
            imgs, targets = data
            outputs = Lr(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("Total loss is{}".format()(total_test_loss))
    print("Total accuracy rate is{}".format()(total_accuracy / test_num))
    writer.add_scalar("test_loss", total_test_loss, test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_num, test_step)

    torch.save(Lr, "Lr_{}.pt".format(i))
    print("Module saved!")

writer.close()