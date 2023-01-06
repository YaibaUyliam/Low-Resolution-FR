import os
import PIL
import logging
import argparse
import numpy as np

import torch
from torch.optim import SGD
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from trainer import Trainer
from vggface2_data_manager import VGGFace2DataManager
from loss import CenterLoss

parser = argparse.ArgumentParser("CR-FR")
# Generic usage
parser.add_argument('-s', '--seed', type=int, default=41, 
                help='Set random seed (default: 41)')
# Model related options
parser.add_argument('-bp', '--model-base-path', default='pretrained/senet50_ft_pytorch.pt', 
                help='Path to base model checkpoint')
parser.add_argument('-ckp', '--model-ckp', 
                help='Path to fine tuned model checkpoint')
parser.add_argument('-ep', '--experimental-path', default='experiments_results', 
                help='Output main path')
parser.add_argument('-tp', '--tensorboard-path', default='experiments_results', 
                help='Tensorboard main log dir path')
# Training Options
parser.add_argument('-dp', '--dset-base-path', default='/home/iu_yaiba/Documents/dataset_fr',
                help='Base path to datasets')
parser.add_argument('-l', '--lambda_', default=0.1, type=float,
                help='Lambda for features regression loss (default: 0.1)')
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, 
                help='Learning rate (default: 1.e-3)')
parser.add_argument('-m', '--momentum', default=0.9, type=float, 
                help='Optimizer momentum (default: 0.9)')
parser.add_argument('-nt', '--nesterov', action='store_true',
                help='Use Nesterov (default: False)')
parser.add_argument('-lp', '--downsampling-prob', default=0.1, type=float,
                help='Downsampling probability (default: 0.1)')
parser.add_argument('-e', '--epochs', type=int, default=1, help='Training epochs (default: 1)')
parser.add_argument('-rs', '--train-steps', type=int, default=1,
                help='Set number of training iterations before each validation run (default: 1)')
parser.add_argument('-c', '--curriculum', action='store_true', 
                help='Use curriculum learning (default: False)')
parser.add_argument('-cs', '--curr-step-iterations', type=int, default=35000, 
                help='Number of images for each curriculum step (default: 35000)')
parser.add_argument('-sp', '--scheduler-patience', type=int, default=10, 
                help='Scheduler patience (default: 10)')
parser.add_argument('-b', '--batch-size', type=int, default=32, 
                help='Batch size (default: 32)')
parser.add_argument('-ba', '--batch-accumulation', type=int, default=8, 
                help='Batch accumulation iterations (default: 8)')
parser.add_argument('-fr', '--valid-fix-resolution', type=int, default=8, 
                help='Resolution on validation images (default: 8)')
parser.add_argument('-nw', '--num-workers', type=int, default=8, 
                help='Number of workers (default: 8)')
args = parser.parse_args()


# ----------------------------- GENERAL ----------------------------------------
tmp = (
    f"{args.lambda_}-{args.learning_rate}-{args.downsampling_prob}-"
    f"{args.train_steps}-{args.curriculum}-{args.curr_step_iterations}"
)

out_dir = os.path.join(args.experimental_path, tmp)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(out_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info(f"Training outputs will be saved at: {out_dir}")
# ------------------------------------------------------------------------------


# --------------------------- CUDA SET UP --------------------------------------
cudnn.benchmark = True

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
# ------------------------------------------------------------------------------


# ---------------- LOAD MODEL & OPTIMIZER & SCHEDULER --------------------------
sm, tm = load_models(args.model_base_path, device, args.model_ckp)
center_loss = CenterLoss(num_classes=2, feat_dim=8631, use_gpu=True)  #Se-resnet-50
params = list(sm.parameters()) + list(center_loss.parameters())

optimizer = SGD(
            params=params, 
            lr=args.learning_rate, 
            momentum=args.momentum, 
            weight_decay=5e-04, 
            nesterov=args.nesterov
        )

scheduler = ReduceLROnPlateau(
                        optimizer=optimizer, 
                        mode='min', 
                        factor=0.5,
                        patience=args.scheduler_patience, 
                        verbose=True,
                        min_lr=1.e-5, 
                        threshold=0.1
                    )
# ------------------------------------------------------------------------------


# ---------------------------- LOAD DATA ---------------------------------------
kwargs = {
    'batch_size': args.batch_size,
    'downsampling_prob': args.downsampling_prob,
    'curriculum': args.curriculum,
    'curr_step_iterations': args.curr_step_iterations, 
    'algo_name': 'bilinear',
    'algo_val': PIL.Image.BILINEAR,
    'valid_fix_resolution': args.valid_fix_resolution,
    'num_of_workers': args.num_workers
}
data_manager = VGGFace2DataManager(
                            dataset_path=args.dset_base_path,
                            img_folders=['images', 'images'],
                            transforms=[get_transforms(mode='train'), get_transforms(mode='eval')],
                            device=device,
                            logging=logging,
                            **kwargs
                        )
a, b = data_manager.get_loaders()

for i in a:
    print(i[2])

# # # ------------------------------------------------------------------------------
# # import cv2 
# # if __name__ == '__main__':
# #     loaders = data_manager.get_loaders()
# #     train_data = loaders[0]

# #     for idx, data_batch in enumerate(train_data, 1):
# #         img_origin = data_batch[1]
# #         img_l = data_batch[0]
# #         print(img_origin[3].shape)
# #         img_origin[3] = img_origin[3].numpy().transpose((1, 2, 0))
# #         cv2.imshow('h', img_origin[3])
# #         cv2.waitKey(0)
# #         cv2.destroyAllWindows()
# #         cv2.imshow('h', img_l[3])
# #         cv2.waitKey(0)
# #         cv2.destroyAllWindows()
# #         break

# from torch.nn import MSELoss, PairwiseDistance
# import torch

# # dis = PairwiseDistance(eps=0)
# # loss = MSELoss()
# # input = torch.randn(1, 2, 3)

# # target = torch.randn(1, 2, 3)

# # print(dis(input, target))

# # a = torch.randn(24, 2048)
# # a = a.unsqueeze(2)
# # print(a.shape)

# import os 
# import glob 
# import random
# import shutil
# list_dir = glob.glob('/home/crist_tienngoc/TOMO/fr/dataset/val/*')
# print(len(list_dir))
# path_val = '/home/crist_tienngoc/TOMO/fr/dataset/val'
# random.shuffle(list_dir)
# count = 0

# for i in list_dir:
#     name_class = os.path.basename(os.path.dirname(i))
#     if not os.path.exists(os.path.join(path_val, name_class)):
#         os.mkdir(os.path.join(path_val, name_class))
    
#     if len(os.listdir(os.path.join(path_val, name_class))) < 6:
#         count += 1
#         shutil.move(i, os.path.join(path_val, name_class))
