import os
import PIL
import logging
import argparse
import numpy as np

import torch
from torch.optim import SGD
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter

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
parser.add_argument('-dp', '--dset-base-path', 
                help='Base path to datasets')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, 
                help='Learning rate (default: 1.e-2)')
parser.add_argument('-m', '--momentum', default=0.9, type=float, 
                help='Optimizer momentum (default: 0.9)')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Training epochs (default: 1)')
parser.add_argument('-rs', '--train-steps', type=int, default=500,
                help='Set number of training iterations before each validation run (default: 1)')
parser.add_argument('-sp', '--scheduler-patience', type=int, default=5, 
                help='Scheduler patience (default: 5)')
parser.add_argument('-b', '--batch-size', type=int, default=16, 
                help='Batch size (default: 32)')
parser.add_argument('-ba', '--batch-accumulation', type=int, default=1,  ## 8 batch moi update parameter mot lan 
                help='Batch accumulation iterations (default: 1)')
parser.add_argument('-nw', '--num-workers', type=int, default=8, 
                help='Number of workers (default: 8)')
parser.add_argument('-nt', '--nesterov', action='store_true',
                help='Use Nesterov (default: False)')
parser.add_argument('-fr', '--valid-fix-resolution', type=int, default=8, 
                help='Resolution on validation images (default: 8)')
args = parser.parse_args()


# ----------------------------- GENERAL ----------------------------------------
tmp = (
    f"{args.learning_rate}--{args.train_steps}"
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

tb_writer = SummaryWriter(os.path.join(args.tensorboard_path, 'tb_runs', tmp))

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
sm, tm = load_models(args.model_base_path, device, args.model_ckp, number_class=10572)

center_loss = CenterLoss(num_classes=2, feat_dim=2048, use_gpu=True)  #Se-resnet-50
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
    'algo_name': 'bilinear',
    'algo_val': PIL.Image.BILINEAR,
    'num_of_workers': args.num_workers,
    'valid_fix_resolution': args.valid_fix_resolution
}
data_manager = VGGFace2DataManager(
                            dataset_path=args.dset_base_path,
                            img_folders=['images', 'images'],
                            transforms=[get_transforms(mode='train'), get_transforms(mode='eval')],
                            device=device,
                            logging=logging,
                            **kwargs
                        )
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    Trainer(
        student=sm, 
        teacher=tm, 
        center_loss=center_loss, 
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=data_manager.get_loaders(),
        device=device,
        batch_accumulation=args.batch_accumulation,
        train_steps=args.train_steps,
        out_dir=out_dir,
        tb_writer=tb_writer,
        logging=logging
    ).train(args.epochs)
