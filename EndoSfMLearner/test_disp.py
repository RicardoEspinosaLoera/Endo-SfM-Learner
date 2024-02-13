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

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument('--resnet-layers', required=False, type=int, default=18, choices=[18, 50], help='depth network architecture.')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CMAP = 'plasma'

def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h,w,_ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img

def _gray2rgb(im, cmap=CMAP):
  cmap = plt.get_cmap(cmap)
  rgba_img = cmap(im.astype(np.float32))
  rgb_img = np.delete(rgba_img, 3, 2)
  return rgb_img


def _normalize_depth_for_display(depth,
                                 pc=95,
                                 crop_percent=0,
                                 normalizer=None,
                                 cmap=CMAP):
  """Converts a depth map to an RGB image."""
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
  return disp

@torch.no_grad()
def main():
    args = parser.parse_args()

    disp_net = models.DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    print(dataset_dir)

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
    else:
        test_files=sorted(dataset_dir.files('*.jpg'))

    print('{} files to test'.format(len(test_files)))
  
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    avg_time = 0
    for j in tqdm(range(len(test_files))):
        tgt_img = load_tensor_image(test_files[j], args)
        # tgt_img = load_tensor_image( dataset_dir + test_files[j], args)

        # compute speed
        torch.cuda.synchronize()
        t_start = time.time()

        output = disp_net(tgt_img)

        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        
        avg_time += elapsed_time

        pred_disp = output.cpu().numpy()[0,0]
        #print(pred_disp.shape)
        depth_map = np.squeeze(pred_disp)
        colored_map = _normalize_depth_for_display(depth_map, cmap=CMAP,normalizer=True)
        print(colored_map.shape)
        imageio.imsave(output_dir/str(j)+'.jpg', colored_map)

        if j == 0:
            predictions = np.zeros((len(test_files), *pred_disp.shape))
        predictions[j] = 1/pred_disp
    
    np.save(output_dir/'predictions.npy', predictions)

    avg_time /= len(test_files)
    print('Avg Time: ', avg_time, ' seconds.')
    print('Avg Speed: ', 1.0 / avg_time, ' fps')


if __name__ == '__main__':
    main()
