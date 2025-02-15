import torch
from skimage.transform import resize as imresize
from imageio import imread
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import models
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=320, type=int, help="Image width")
parser.add_argument("--min-depth", default=0.1)
parser.add_argument("--max-depth", default=150.0)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument('--resnet-layers', required=False, type=int, default=18, choices=[18, 50], help='depth network architecture.')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#CMAP = 'plasma'
_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

def load_tensor_image(filename, args):
    #img = imread(filename).astype(np.float32)
    #img = Image.open("D:\\Phd_Research\\Datasets\\SERV-CT\\SERV-CT\\"+filename)
    img = Image.open("D:\\Phd_Research\\Datasets\\Hamlyn\\"+filename)
    img = np.array(img).astype(np.float32)
    h,w,_ = img.shape
    #print("Heeeere",h,w)
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img

def colormap(inputs, normalize=True, torch_transpose=True):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()

        vis = inputs
        if normalize:
            ma = float(vis.max())
            mi = float(vis.min())
            d = ma - mi if ma != mi else 1e5
            vis = (vis - mi) / d

        if vis.ndim == 4:
            vis = vis.transpose([0, 2, 3, 1])
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[:, :, :, 0, :3]
            if torch_transpose:
                vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 3:
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[:, :, :, :3]
            if torch_transpose:
                vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 2:
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[..., :3]
            if torch_transpose:
                vis = vis.transpose(2, 0, 1)

        return vis
"""
def _gray2rgb(im, cmap=CMAP):
  cmap = plt.get_cmap(cmap)
  rgba_img = cmap(im.astype(np.float32))
  rgb_img = np.delete(rgba_img, 3, 2)
  return rgb_img"""

"""
def _normalize_depth_for_display(depth,
                                 pc=95,
                                 crop_percent=0,
                                 normalizer=None,
                                 cmap=CMAP):
  # Convert to disparity.
  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  disp = _gray2rgb(disp, cmap=cmap)
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp"""

@torch.no_grad()
def main():
    args = parser.parse_args()

    disp_net = models.DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    print(dataset_dir)

    #if args.dataset_list is not None:
    with open("hamlyn_test.txt", 'r') as f:
        test_files = list(f.read().splitlines())
        print(test_files)
    #else:
        #test_files=dataset_dir.files('*.jpg')
        #print("Sorted")

    print('{} files to test'.format(len(test_files)))
  
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    avg_time = 0
    for j in tqdm(range(len(test_files))):
        tgt_img = load_tensor_image(test_files[j], args)
        

        # compute speed
        torch.cuda.synchronize()
        t_start = time.time()

        output = disp_net(tgt_img)

        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        
        avg_time += elapsed_time

        pred_disp = output.cpu().numpy()[0,0]
        
        print(pred_disp.shape)
        #depth_map = np.squeeze(pred_disp)
        #colored_map = _normalize_depth_for_display(depth_map, cmap=CMAP,normalizer=True)
        #print(colored_map.shape)
        
        disp = colormap(pred_disp)
        disp = (disp * 255).astype(np.uint8)
        #print(disp)
        im = Image.fromarray(disp.transpose(1, 2, 0))
        imageio.imsave(output_dir/str(j)+'.jpg', im)

        if j == 0:
            predictions = np.zeros((len(test_files), *pred_disp.shape))
        predictions[j] = 1/pred_disp
    
    np.save(output_dir/'predictions.npy', predictions)

    avg_time /= len(test_files)
    print('Avg Time: ', avg_time, ' seconds.')
    print('Avg Speed: ', 1.0 / avg_time, ' fps')


if __name__ == '__main__':
    main()


