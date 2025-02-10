from basic_fcn2 import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import multiprocessing
import torch.optim as optim

num_workers = multiprocessing.cpu_count()

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



def getClassWeights(train_dataset):
    """Calculate class weights using median frequency balancing.
    
    Args:
        train_dataset: A PyTorch Dataset containing training data.
        
    Returns:
        A torch.Tensor containing weights for each of the 21 classes.
    """
    num_classes = 21
    class_counts = torch.zeros(num_classes)
    total_pixels = 0
    
    # Count frequencies per class, ignoring pixels with label 255.
    for _, target in train_dataset:
        target_np = np.array(target, dtype=np.int32)
        total_pixels += target_np.size
        for c in range(num_classes):
            class_counts[c] += torch.sum(target == c)

    # Frequency for each class
    freq = class_counts / (total_pixels + 1e-10)
    freq_sum = torch.sum(freq)
    
    # Compute weights: 1 - (class_freq / freq_sum)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        weights[c] = 1 - (freq[c] / freq_sum)
    
    return weights

# normalize using imagenet averages
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

target_transform = MaskToTensor()

# train_transformed = voc.VOC('train', transform=input_transform, target_transform=target_transform, modification=True)
train_original = voc.VOC('train', transform=input_transform, target_transform=target_transform)
# train_dataset = torch.utils.data.ConcatDataset([train_transformed, train_original])
valtest_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)


# Split valtest_dataset into val_dataset and test_dataset
val_size = len(valtest_dataset)//2
test_size = len(valtest_dataset) - val_size
val_dataset, test_dataset = torch.utils.data.random_split(valtest_dataset, [val_size, test_size])

train_loader = DataLoader(dataset=train_original, batch_size= 16, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)

epochs = 30
n_class = 21

fcn_model = UNet(in_channels=3, out_channels=n_class)
fcn_model.apply(init_weights)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# class_weights = getClassWeights(train_dataset).to(device)

optimizer = optim.SGD(fcn_model.parameters(), lr=0.01, momentum=0.9)

#class_weights = getClassWeights(train_original).to(device)
criterion = nn.CrossEntropyLoss()

# eta_min = 0.001
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*2, eta_min=0.0005)
fcn_model = fcn_model.to(device)

def train():
    """
    Train a deep learning model using mini-batches.

    Returns:
        None.
    """

    best_iou_score = 0.0
    best_model_path = "best_model.pth"
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement before stopping
    epochs_no_improve = 0

    for epoch in range(epochs):
        ts = time.time()
        fcn_model.train()  # Set the model to training mode

        for iter, (inputs, labels) in enumerate(train_loader):
            # Reset optimizer gradients
            optimizer.zero_grad()

            # Transfer the input and labels to the same device as the model's
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute outputs
            outputs = fcn_model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagate
            loss.backward()

            # Update model weights
            optimizer.step()

            if iter % 20 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        scheduler.step()  # Update learning rate
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        # Evaluate model on validation set
        current_iou_score, val_loss = val(epoch)

        # Save model state if IoU score improves
        if current_iou_score > best_iou_score:
            best_iou_score = current_iou_score
            torch.save(fcn_model.state_dict(), best_model_path)
            print(f"Model saved with mIoU: {best_iou_score:.4f}")

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("Training completed.")

def val(epoch):
    """
    Validate the deep learning model on a validation dataset.

    Args:
        epoch (int): The current epoch number.

    Returns:
        tuple: Mean IoU score and mean loss for this validation epoch.
    """
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            mean_iou = util.iou(outputs, labels, n_classes = n_class)
            mean_iou_scores.append(mean_iou)

            acc = util.pixel_acc(torch.argmax(outputs, dim=1), labels)
            accuracy.append(acc)


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    return np.mean(mean_iou_scores), np.mean(losses)

def modelTest():
    """
    Test the deep learning model using a test dataset.

    Returns:
        None. Outputs average test metrics to the console.
    """

    fcn_model.load_state_dict(torch.load('best_model.pth'))
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !



    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        all_accuracy = []
        all_iou = []
        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model(inputs)

            loss = criterion(outputs, labels)

            mean_iou = util.iou(torch.argmax(outputs, dim=1), labels)
            all_iou.append(mean_iou)
            accuracy = util.pixel_acc(torch.argmax(outputs, dim=1), labels)
            all_accuracy.append(accuracy)

            print(f"Iteration: {iter}, Test Loss: {loss.item()}, Test IoU: {mean_iou}, Test Pixel Acc: {accuracy}")

        average_accuracy = np.mean(all_accuracy)
        average_iou = np.mean(all_iou)
        print(f"Average Test Pixel Accuracy: {average_accuracy}")
        print(f"Average Test IoU: {average_iou}")

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


def exportModel(inputs):    
    """
    Export the output of the model for given inputs.

    Args:
        inputs: Input data to the model.

    Returns:
        Output from the model for the given inputs.
    """

    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    saved_model_path = "best_model.pth"
    fcn_model.load_state_dict(torch.load(saved_model_path))  # Load the best model weights
    
    inputs = inputs.to(device)
    
    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
        output_image = fcn_model(inputs)
    
    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return output_image

if __name__ == "__main__":

    val(0)  # show the accuracy before training
    train()
    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()

