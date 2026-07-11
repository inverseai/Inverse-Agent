import torch


def split_and_normalize(features):
    train, validation = torch.utils.data.random_split(features, [800, 200])
    train_tensor = torch.stack(list(train))
    validation_tensor = torch.stack(list(validation))
    mean = train_tensor.mean(dim=0)
    std = train_tensor.std(dim=0)
    return (train_tensor - mean) / std, (validation_tensor - mean) / std


def evaluate(model, loader):
    was_training = model.training
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            correct += (model(inputs).argmax(dim=1) == targets).sum().item()
    if was_training:
        model.train()
    return correct / len(loader.dataset)
