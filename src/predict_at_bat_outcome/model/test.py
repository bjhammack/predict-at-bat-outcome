import torch


def evaluate(model, loss_func, test_dl):
    model.eval()

    running_loss = 0.0
    correct_labels = 0
    total_labels = 0
    predictions = []
    probabilities = []
    labels = []
    
    with torch.no_grad():
        for data in test_dl:
            inputs, label = data
            outputs = model(inputs)
            probability, prediction = torch.max(outputs, 1)

            predictions.append(prediction.item())
            probabilities.append(probability.item())
            labels.append(label.item())

            loss = loss_func(outputs, label).item()
            running_loss += loss

            correct_labels += (prediction == label).sum().item()
            total_labels += prediction.shape[0]

        acc = correct_labels / total_labels
        return {
            'accuracy': acc,
            'loss': loss,
            'predictions': predictions,
            'probabilities': probabilities,
            'labels':labels
            }


# def create_pr_curve(class_index, test_probs, test_label, global_step=0):
#     '''
#     Takes in a "class_index" from 0 to 9 and plots the corresponding
#     precision-recall curve
#     '''
#     tensorboard_truth = test_label == class_index
#     tensorboard_probs = test_probs[:, class_index]

#     writer.add_pr_curve(classes[class_index],
#                         tensorboard_truth,
#                         tensorboard_probs,
#                         global_step=global_step)
#     writer.close()