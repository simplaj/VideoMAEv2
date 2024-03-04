import torch
import torch.onnx
import models
from decord import VideoReader, cpu
from timm.models import create_model
from dataset import video_transforms, volume_transforms
import numpy as np


data_resize = video_transforms.Compose([
    video_transforms.Resize(
        size=(224), interpolation='bilinear')
])
data_transform = video_transforms.Compose([
    volume_transforms.ClipToTensor(),
    video_transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def model_load(mode_path):
    model = create_model(
        'vit_base_patch16_224',
        img_size=224,
        pretrained=False,
        num_classes=3,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True
    )
    ckp = torch.load(model_path, map_location='cpu')
    ckp_model = ckp['model']
    model.load_state_dict(ckp_model)
    return model
    

def video_read(
    video_path,
    clip_len=16,
    sparse_sample=False,
    frame_sample_rate=4,
    test_num_crop=2,
    test_num_segment=3,
    chunk_nb=0,
    split_nb=0,
    ):
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    length = len(vr)
    if not sparse_sample:
        all_index = [
            x for x in range(0, length, frame_sample_rate)
        ]
        while len(all_index) < clip_len:
            all_index.append(all_index[-1])
            
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    
    buffer = data_resize(buffer)
    if isinstance(buffer, list):
        buffer = np.stack(buffer, 0)
    
    spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                          224) / (test_num_crop - 1)
    temporal_step = max(
        1.0 * (buffer.shape[0] - clip_len) / 
        (test_num_segment - 1), 0
    )
    temporal_start = int(chunk_nb * temporal_step)
    spatial_start = int(split_nb * spatial_step)
    if buffer.shape[1] >= buffer.shape[2]:
        buffer = buffer[temporal_start:temporal_start +
                        clip_len,
                        spatial_start:spatial_start +
                        224, :, :]
    else:
        buffer = buffer[temporal_start:temporal_start +
                        clip_len, :,
                        spatial_start:spatial_start +
                        224, :]
    buffer = data_transform(buffer)
    return buffer


if __name__ == '__main__':
    export = True
    model_path = '/root/autodl-tmp/train_results/v2/vit_b_pd_ft_weight_7e-4_lr_240304/checkpoint-19.pth'
    video_path = '/root/autodl-tmp/videos_all/test/mild/ill_8_2.mp4'
    model = model_load(model_path)
    model.eval()
    data = video_read(video_path)
    data = data.unsqueeze(0)
    with torch.no_grad():
        output = model(data)
        print(output)
    if export:
        torch.onnx.export(
            model,
            data,
            'videomaev2.onnx'
        )