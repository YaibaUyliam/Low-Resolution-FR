import os

import torch
import torch.nn as nn
import torchvision.transforms as t
from facenet_pytorch import InceptionResnetV1
import numpy as np 

def load_models(model_base_path, device="cpu", model_ckp=None, number_class=2):
    assert os.path.exists(model_base_path), "Base model checkpoint not found at: {}".format(model_base_path)
    # sm = torch.load(model_base_path)
    # tm = torch.load(model_base_path)
    sm = InceptionResnetV1(classify=True, num_classes=number_class)
    tm = InceptionResnetV1(classify=True, num_classes=number_class)
    tm.load_state_dict(torch.load('/home/crist_tienngoc/TOMO/fr/train_teacher/weights/weight_5'))
    sm.load_state_dict(torch.load('/home/crist_tienngoc/TOMO/fr/train_teacher/weights/weight_5'))
    
    # if model_ckp is not None:
    #     assert os.path.exists(model_ckp), f"Model checkpoint not found at: {model_ckp}"
    #     ckp = torch.load(model_ckp, map_location='cpu')
    #     [p.data.copy_(torch.from_numpy(ckp['model_state_dict'][n].numpy())) for n, p in sm.named_parameters()]
    #     for n, m in sm.named_modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.momentum = 0.1
    #             m.running_var = ckp['model_state_dict'][n + '.running_var']
    #             m.running_mean = ckp['model_state_dict'][n + '.running_mean']
    #             m.num_batches_tracked = ckp['model_state_dict'][n + '.num_batches_tracked']

    # sm.classifier_1 = nn.Linear(in_features = 2048, out_features = number_class, bias = True)
    # tm.classifier_1 = nn.Linear(in_features = 2048, out_features = number_class, bias = True)
    # ## Freeze all params for the teacher model
    for param in tm.parameters():
        param.requires_grad = False
    return sm.to(device), tm.to(device)


def save_model_checkpoint(best_acc, batch_idx, epoch, model_state_dict, out_dir, logging):
    # state_dict = {
    #             'best_acc': best_acc,
    #             'epoch': epoch,
    #             'model_state_dict': model_state_dict
    #         }
    file_name = os.path.join(out_dir, f'models_ckp_{epoch}_{batch_idx}.pth')
    torch.save(model_state_dict, file_name)
    #torch.save(state_dict, file_name)
    logging.info(
            f"Saved model with best acc: {best_acc}"
            f"\nAt epoch: {epoch}"
            f"\nAt iter: {batch_idx}"
            f"\nModel saved at: {file_name}"
        )

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def get_transforms(mode, resize=256, grayed_prob=0.2, crop_size=224):
    # def subtract_mean(x):
    #     mean_vector = [91.4953, 103.8827, 131.0912]
    #     x *= 255.
    #     x[0] -= mean_vector[0]
    #     x[1] -= mean_vector[1]
    #     x[2] -= mean_vector[2]
    #     return x
    if mode=='train':
        return t.Compose([
                    #t.Resize(resize),
                    t.RandomGrayscale(p=grayed_prob),
                    #t.RandomCrop(crop_size),
                    t.ToTensor(),
                    fixed_image_standardization
                    #t.Lambda(lambda x: subtract_mean(x))
                ])
    else:
        return t.Compose([
                    #t.Resize(resize),
                    #t.CenterCrop(crop_size),
                    t.ToTensor(),
                    fixed_image_standardization
                    #t.Lambda(lambda x: subtract_mean(x))
                ])

