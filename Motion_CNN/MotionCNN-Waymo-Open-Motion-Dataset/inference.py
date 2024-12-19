import torch
import timm
from train import MotionCNNDataset, dict_to_cuda, postprocess_predictions
from torch.utils.data import Dataset, DataLoader
from utils import get_config
from losses import NLLGaussian2d
import numpy as np

def model_setup():
    n_components = 5
    n_modes = 3
    n_timestamps = 80
    output_dim = n_modes + n_modes * n_timestamps * n_components
    model = timm.create_model(
            'resnet34', pretrained=True,
            in_chans=27, num_classes=output_dim)

    checkpoint_data = torch.load('/data/cmpe258-sp24/010892622/MotionCNN-Waymo-Open-Motion-Dataset/model_checkpoints/basic/e2_b40000.pth')
    model.load_state_dict(checkpoint_data['model_state_dict'])

    return model

def data_setup():
    general_config = get_config('/data/cmpe258-sp24/010892622/MotionCNN-Waymo-Open-Motion-Dataset/configs/basic.yaml')
    training_config = general_config['training']
    model_config = general_config['model']

    testing_dataloader = DataLoader(
            MotionCNNDataset('/data/cmpe258-sp24/010892622/data/waymo_motion_prediction_v1.0.0_tf/processed_tf_example/testing', load_roadgraph=True),
            **training_config['val_dataloader'])
    return testing_dataloader, model_config

def test():
    model = model_setup()
    testing_dataloader, model_config = data_setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    loss_module = NLLGaussian2d()
    test_loss = []
    with torch.no_grad(): 
        for inputs in testing_dataloader:
            test_data = dict_to_cuda(inputs)
            prediction_tensor = model(test_data['raster'].float())
            prediction_dict = postprocess_predictions(prediction_tensor, model_config)
            loss = loss_module(test_data, prediction_dict)
            test_loss.append(loss.item())
            
    
    mean_test_loss = np.mean(test_loss)
    print(f"Mean Test Loss: {mean_test_loss:.3f}")

print("Starting Inference for Motion CNN")
test()
print("Inference Complete")