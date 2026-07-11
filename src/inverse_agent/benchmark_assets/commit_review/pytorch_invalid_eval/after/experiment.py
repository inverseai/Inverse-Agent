import torch


def split_and_normalize(features):
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    normalized = (features - mean) / std
    train, validation = torch.utils.data.random_split(normalized, [800, 200])
    train_tensor = torch.stack(list(train))
    validation_tensor = torch.stack(list(validation))
    return train_tensor, validation_tensor


def evaluate(model, loader):
    model.train()
    correct = 0
    for inputs, targets in loader:
        correct += (model(inputs).argmax(dim=1) == targets).sum().item()
    return correct / len(loader.dataset)
