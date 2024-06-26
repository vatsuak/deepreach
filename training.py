'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
import time
import numpy as np
import os
import shutil
import json


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, loss_fn_val,
          summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          validation_fn=None, start_epoch=0, args=None, adjust_relative_grads=False):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    # Load the checkpoint if required
    if start_epoch > 0:
        # Load the model and start training from that point onwards
        model_path = os.path.join(model_dir, 'checkpoints', 'model_epoch_%04d.pth' % start_epoch)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.train()
        optim.load_state_dict(checkpoint['optimizer'])
        optim.param_groups[0]['lr'] = lr
        assert(start_epoch == checkpoint['epoch'])
    else:
        # Start training from scratch
        if os.path.exists(model_dir):
            val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
            if val == 'y':
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        opt_path = os.path.join(model_dir, 'commandline_args.txt')
        with open(opt_path, "w") as fp:
            json.dump(vars(args),fp, indent=2) 

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0

    if adjust_relative_grads:
        new_weight1 = 1
        new_weight2 = 1

    with tqdm(total=epochs) as pbar:
        train_losses = []
        for epoch in range(start_epoch, epochs):
            if not epoch % epochs_til_checkpoint:
                # Saving the optimizer state is important to produce consistent results
                checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict()}
                torch.save(checkpoint,
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                # torch.save(model.state_dict(),
                #            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
                if validation_fn is not None:
                    validation_fn(model, checkpoints_dir, epoch)

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                # import pdb;pdb.set_trace()
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)
                
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                # Adjust the relative magnitude of the losses if required
                if adjust_relative_grads:
                    if losses['diff_constraint_hom'] > 0.01 and losses['dirichlet'] > 0.01: #To make sure not getting nans when the dirichlet loss is zero
                    # if losses['diff_constraint_hom'] > 0.01:
                        params = OrderedDict(model.named_parameters())
                        # Gradients with respect to the PDE loss
                        optim.zero_grad()
                        losses['diff_constraint_hom'].backward(retain_graph=True)
                        grads_PDE = []
                        for key, param in params.items():
                            grads_PDE.append(param.grad.view(-1))
                        grads_PDE = torch.cat(grads_PDE)

                        # Gradients with respect to the boundary loss
                        optim.zero_grad()
                        losses['dirichlet'].backward(retain_graph=True)
                        grads_dirichlet = []
                        for key, param in params.items():
                            grads_dirichlet.append(param.grad.view(-1))
                        grads_dirichlet = torch.cat(grads_dirichlet)

                        # Set the new weight according to the paper
                        num = torch.mean(torch.abs(grads_PDE))
                        den = torch.mean(torch.abs(grads_dirichlet))
                        new_weight1 = 0.9*new_weight1 + 0.1*num/den
                        losses['dirichlet'] = new_weight1 * losses['dirichlet']
                        writer.add_scalar('weight_scaling1', new_weight1, total_steps)

                        # # Gradients with respect to the switching loss
                        # optim.zero_grad()
                        # losses['switching_loss'].backward(retain_graph=True)
                        # grads_switching = []
                        # for key, param in params.items():
                        #     grads_switching.append(param.grad.view(-1))
                        # grads_switching = torch.cat(grads_switching)

                        # # Gradients with respect to the periodicity loss
                        # if 'periodicity' in losses.keys():
                        #     optim.zero_grad()
                        #     losses['periodicity'].backward(retain_graph=True)
                        #     grads_periodicity = []
                        #     for key, param in params.items():
                        #         grads_periodicity.append(param.grad.view(-1))
                        #     grads_periodicity = torch.cat(grads_periodicity)

                        #     # Set the new weight according to the paper
                        #     num = torch.mean(torch.abs(grads_PDE))
                        #     den = torch.mean(torch.abs(grads_periodicity))
                        #     new_weight2 = 0.9*new_weight2 + 0.1*num/den
                        #     losses['periodicity'] = new_weight2 * losses['periodicity']
                        #     writer.add_scalar('weight_scaling2', new_weight2, total_steps)

                        # import ipdb; ipdb.set_trace()

                # import ipdb; ipdb.set_trace()

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    # tqdm.write("Epoch %d, Total loss %0.6f, CurrTrainingTime %0.2fs, Counter %d, Total Count %d, \
                    #             iteration time %0.6f" % (epoch, train_loss, gt['horizon'], gt['counter'], gt['full_count'], time.time() - start_time))
                    tqdm.write("Epoch %d, Total loss %0.6f, Dirichlet loss %0.6f, PDE loss %0.6f CurrTrainingTime %0.2fs, Counter %d, Total Count %d, SeqStart %0.2f SeqEnd %0.2f" % (epoch, train_loss, losses['dirichlet'], losses['diff_constraint_hom'],
                                                                        gt['horizon'], gt['counter'], gt['full_count'], gt["SeqStart"],gt["SeqEnd"]))
                                                                        
                    utils.weight_histograms(writer, epoch, model)
                    if val_dataloader is not None:
                        # print("Running validation set...")
                        model.eval()
                        # with torch.no_grad():
                        val_loss = 0
                        for (model_input, gt) in val_dataloader:
                            model_output = model(model_input)
                            # import pdb;pdb.set_trace()
                            val_losses = loss_fn_val(model_output, gt)
                            for loss_name, loss in val_losses.items():
                                single_loss = loss.mean()
                                writer.add_scalar(loss_name+"_validation", single_loss, total_steps)
                                val_loss += single_loss
                       
                        writer.add_scalar("total_validation_loss", train_loss, total_steps)
                        optim.zero_grad()
                        model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
