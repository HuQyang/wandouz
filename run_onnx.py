import argparse
import torch
from douzero.dmc.models import Model

def export_model(position, checkpoint, dummy_obs_z, dummy_obs_x, output_onnx, opset):
    model_instance = Model(device='cpu')
    m = model_instance.get_model(position)
    print(checkpoint["stats"])
    m.load_state_dict(checkpoint["model_state_dict"][position])
    m.eval()
    
    torch.onnx.export(
        m,
        (dummy_obs_z, dummy_obs_x),
        output_onnx,
        opset_version=opset,
        input_names=["obs_z", "obs_x"],
        output_names=["output"],
        dynamic_axes={"obs_z": {0: "batch_size"}, "obs_x": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Exported {position} to {output_onnx}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export trained models to ONNX")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    args = parser.parse_args()

    day = 28
    if day == 7:
        ite = 504112000 # 7 days
    elif day == 14:
        ite = 773708800
    elif day == 21:
        ite = 1342233600
    else:
        ite = 1952918400


    landlord_ckpt = f'douzero_checkpoints/douzero/landlord_weights_{ite}.ckpt'
    landlord_up_ckpt = f'douzero_checkpoints/douzero/landlord_up_weights_{ite}.ckpt'
    landlord_down_ckpt = f'douzero_checkpoints/douzero/landlord_down_weights_{ite}.ckpt'
    
    landlord_ckpt = 'douzero_checkpoints/douzero/model.tar'
    
    batch_size = 32
    dummy_obs_z = torch.randn(batch_size, 5, 201)  # for example: batch_size=1, feature_dim=10

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        output_onnx = f"{position}_day{day}.onnx"
        if position == 'landlord':
            landlord_checkpoint = torch.load(landlord_ckpt, map_location='cpu')
            dummy_obs_x = torch.randn(batch_size, 451)  # for example: batch_size=1, feature_dim=20
        if position == 'landlord_up':
            landlord_checkpoint = torch.load(landlord_ckpt, map_location='cpu')
            dummy_obs_x = torch.randn(batch_size, 588) 
        if position == 'landlord_down':
            landlord_checkpoint = torch.load(landlord_ckpt, map_location='cpu')
            dummy_obs_x = torch.randn(batch_size, 588)
        export_model(position, landlord_checkpoint, dummy_obs_z, dummy_obs_x, output_onnx, args.opset)