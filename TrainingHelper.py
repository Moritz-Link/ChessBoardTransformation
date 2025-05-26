import torch
from torch.utils.data import Dataset,DataLoader
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainDataset = args["TrainDataset"]
    valDataset = args["ValDataset"]
    testDataset = args["TestDataset"]

    TrainDataLoader = DataLoader(trainDataset, batch_size=args["TrainBatchSize"], shuffle=True, drop_last=True)
    ValidDataLoader = DataLoader(valDataset, batch_size=args["ValBatchSize"], shuffle=True, drop_last=True)
    TestDataLoader = DataLoader(testDataset, batch_size=args["TestBatchSize"], shuffle=True, drop_last=True)
    EarlyStopping = args["EarlyStopping"]
    model = args["model"]
    model.to(device)
    LossFunction = args["LossFunction"]
    Optimizer = args["Optimizer"](model.parameters(), lr = args["Optimizer_LR"])

    training_loss = 0

    loss_dict = {
        "train_loss": [],
        "validation_loss_mse": [],
        "validation_loss_mae": [],
    }

    for epoch in range(args["EPOCHS"]):
        epoch_loss_avg = train_one_epoch(TrainDataLoader, LossFunction, Optimizer, model, device)
        #training_loss += epoch_loss_avg
        #Evaluate
        eval_loss, eval_mse, eval_mae = evaluate(ValidDataLoader, model, device, LossFunction)
        loss_dict["train_loss"].append(epoch_loss_avg)
        loss_dict["validation_loss_mse"].append(eval_mse)
        loss_dict["validation_loss_mae"].append(eval_mae)
        if ((epoch+1)% 5 ==0) or (epoch == 0):
          print(f'Epoch {epoch+1}/{args["EPOCHS"]}: Train Loss: {epoch_loss_avg:.4f} | Validation Metrics: Avg Loss: {eval_loss:.4f}, Avg MSE per Keypoint: {eval_mse:.4f}, Avg MAE per Keypoint: {eval_mae:.4f}')

        # early Stopping
        if EarlyStopping.early_stop:
            print("Early stopping")
            break
        else:
            if epoch+1 > args["StartEarlyStopping"]:
              EarlyStopping(eval_loss,model)


    #save Model - in Ealry Stopping normal!

    #training_loss /= args["EPOCHS"]
    return loss_dict


def train_one_epoch(TrainDataLoader, LossFunction, Optimizer, model, device):
    model.to(device)
    model.train()
    epoch_loss = 0
    for images, images_dict in TrainDataLoader:
        Optimizer.zero_grad()
        targets = images_dict["keypoints"].view(images_dict["keypoints"].size(0), -1)
        output = model(images.to(device))
        loss = LossFunction(output, targets.to(device))

        loss.backward()
        Optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(TrainDataLoader)



def evaluate(ValDataLoader, model, device, LossFunction):
    model.to(device)
    model.eval()

    total_loss = 0
    total_mse_per_keypoint = 0
    total_mae_per_keypoint = 0
    num_batches = 0

    with torch.no_grad():
        for images, images_dict in ValDataLoader:
            targets = images_dict["keypoints"].view(images_dict["keypoints"].size(0), -1)
            output = model(images.to(device))

            # Calculate overall loss
            loss = LossFunction(output, targets.to(device))
            total_loss += loss.item()

            # Calculate metrics per keypoint
            # Reshape output and targets to (batch_size, num_keypoints, 2)
            output_reshaped = output.view(output.size(0), -1, 2)
            targets_reshaped = targets.to(device).view(targets.size(0), -1, 2)

            # Calculate squared error per keypoint
            squared_error_per_keypoint = torch.mean((output_reshaped - targets_reshaped)**2, dim=2)
            total_mse_per_keypoint += torch.mean(squared_error_per_keypoint).item()

            # Calculate absolute error per keypoint
            absolute_error_per_keypoint = torch.mean(torch.abs(output_reshaped - targets_reshaped), dim=2)
            total_mae_per_keypoint += torch.mean(absolute_error_per_keypoint).item()

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mse_per_keypoint = total_mse_per_keypoint / num_batches
    avg_mae_per_keypoint = total_mae_per_keypoint / num_batches



    return avg_loss, avg_mse_per_keypoint, avg_mae_per_keypoint


