import torch


def evaluate(model, loss_func, test_dl):
    model.eval()

    running_loss = 0.0
    correct_labels = 0
    total_labels = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in test_dl:
            inputs, label = data
            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)

            predictions.append(prediction.item())
            labels.append(label.item())

            loss = loss_func(outputs, label).item()
            running_loss += loss

            correct_labels += (prediction == label).sum().item()
            total_labels += prediction.shape[0]

        acc = correct_labels / total_labels
        return {'accuracy': acc, 'loss': loss, 'predictions': predictions, 'labels':labels}