import numpy as np
from tqdm import tqdm

import torch.nn.functional as F

from utils import *
from loss import *
class Trainer(object):
    def __init__(self, student, teacher, center_loss, optimizer, scheduler, loaders, device, batch_accumulation, 
                train_steps, out_dir, tb_writer, logging):
        self._student = student
        self._teacher = teacher
        self._center_loss = center_loss
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._train_loader, self._valid_loader_lr = loaders
        self._device = device
        self._batch_accumulation = batch_accumulation
        self._train_steps = train_steps
        self._out_dir = out_dir
        self._tb_writer = tb_writer
        self._logging = logging
        self._it_t = 0
        self._it_v = 0

    def _eval_batch(self, loader_idx, data):
        if loader_idx == 0:  # ImageFolder for original sized images
            batch_original = batch = data[0]
            labels = data[1]
        else:  # custom data set for down sampled images
            batch = data[0]
            batch_original = data[1]
            labels = data[2]
        
            #print('--------------------------------')
            #print(batch)
            #print(labels)
        teacher_features, teacher_logits = self._teacher(batch_original.to(self._device))
        student_features, student_logits = self._student(batch.to(self._device))
        
        #print('.....................', student_logits)
        correct = (student_logits.argmax(dim=1).cpu() == labels).sum().item() 
        #loss_triple = triple_loss(teacher_features.unsqueeze(2), student_features.unsqueeze(2))
        loss_total = F.cross_entropy(student_logits, labels.to(self._device)) + 0.008*self._center_loss(student_features, labels.to(self._device))  \
                                                                              #+ 0.1*loss_triple
        # print('--------------', teacher_features.shape, '------------------')
        # print('--------------', student_features.shape, '------------------')
        # print('--------------', student_logits.shape, '------------------')
        #print('--------------', loss_total, '------------------')
        return loss_total, student_logits, labels, correct

    def _train(self, epoch):
        self._logging.info("#"*30)
        self._logging.info(f'Training at epoch: {epoch}')
        self._logging.info("#"*30)

        self._student.train()
        self._optimizer.zero_grad()

        j = 1
        loss_ = 0
        best_acc = 0
        correct_ = 0
        n_samples_ = 0
        nb_backward_steps = 0

        for batch_idx, data in enumerate(self._train_loader, 1): #enumerate(iter, start=)
            if nb_backward_steps == self._train_steps:
                nb_backward_steps = 0
                v_l_, tmp_best_acc = self._val(epoch)
                self._student.train()
                self._scheduler.step(v_l_, epoch+1)
                ## Save best model
                if tmp_best_acc > best_acc:
                    best_acc = tmp_best_acc
                    save_model_checkpoint(
                                    best_acc, 
                                    batch_idx, 
                                    epoch, 
                                    self._student.state_dict(), 
                                    self._out_dir, 
                                    self._logging
                                )
                
            loss, logits, labels, correct = self._eval_batch(loader_idx=-1, data=data)

            loss_ += loss.item()
            correct_ += correct
            n_samples_ += labels.shape[0]
            loss.backward()
            for param in self._center_loss.parameters():
                param.grad.data *= (1./0.008)

            if j % self._batch_accumulation == 0:
                if nb_backward_steps%10 == 1:
                    self._logging.info(
                                f'Train [{epoch}] - [{batch_idx}]/[{len(self._train_loader)}]:'
                                f'\n\t\t\tLoss LR: {loss_/batch_idx:.3f} --- Acc LR: {(correct_/n_samples_)*100:.2f}%'                               
                            )
                    
                    self._it_t += 1
                    self._tb_writer.add_scalar('train/loss', loss_/batch_idx, self._it_t)
                    self._tb_writer.add_scalar('train/accuracy', correct_/n_samples_, self._it_t)
                
                j = 1
                nb_backward_steps += 1
                self._optimizer.step()
                self._optimizer.zero_grad()

            else:
                j += 1
        
    def _val(self, epoch):
        self._student.eval()
        
        with torch.no_grad():
            #for loader_idx, local_loader in enumerate([self._valid_loader, self._valid_loader_lr]):
                loader_idx = 2
                loss_ = 0.0
                correct_ = 0.0
                n_samples = 0
                desc = 'Validaiont HR' if loader_idx == 0 else 'Validation LR'

                for batch_id, data in enumerate(tqdm(self._valid_loader_lr, total=len(self._valid_loader_lr), desc=desc, leave=False)):
                    loss, logits, labels, correct = self._eval_batch(loader_idx, data)
                    
                    loss_ += loss.item()
                    correct_ += correct
                    n_samples += labels.shape[0]

                loss_ = loss_ / len(self._valid_loader_lr)
                acc_ = (correct_ / n_samples) * 100

                if loader_idx == 0:
                    self._logging.info(f'Valid loss HR: {loss_:.3f} --- Valid acc HR: {acc_:.2f}%')
                    self._tb_writer.add_scalar('validation/loss_hr', loss_, self._it_v)
                    self._tb_writer.add_scalar('validation/accuracy_hr', acc_, self._it_v)
                else:
                    self._logging.info(f'Valid loss LR: {loss_:.3f} --- Valid acc LR: {acc_:.2f}%')
                    self._tb_writer.add_scalar('validation/loss_lr', loss_, self._it_v)
                    self._tb_writer.add_scalar('validation/accuracy_lr', acc_, self._it_v)

        self._it_v += 1

        return loss_, acc_

    def train(self, epochs):
        #self._val(0)
        [self._train(epoch) for epoch in range(1, epochs+1)]

