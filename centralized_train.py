import torch

from fed_baseline import get_model, load_datasets, test, train

# experiments settings
NUM_CLIENTS = 2
BATCH_SIZE = 128
MOMENTUM = 0.9
LEARNIGN_RATE = 1e-2
WEIGHT_DECAY = 1e-4
EPOCHS = 300
SEED = 42
REPEATS = 3
WARMUP = 5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__}")

train_loaders, val_loaders, test_loader = load_datasets(num_clients=1)
train_loader = train_loaders[0]
val_loader = val_loaders[0]

net = get_model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(),
    lr=LEARNIGN_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer=optimizer, start_factor=0.1, total_iters=WARMUP
)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer, milestones=[150, 250], gamma=0.1
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer=optimizer,
    schedulers=[warmup_scheduler, train_scheduler],
    milestones=[WARMUP],
)

for epoch in range(EPOCHS):
    train(
        net,
        train_loader,
        epochs=1,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    loss, acc = test(net, val_loader)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.4f} in Validataion set.")
    test(net, test_loader)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.4f} in Test set.")
