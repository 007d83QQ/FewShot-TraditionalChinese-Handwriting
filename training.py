import os
import sys
import time

import hydra
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import random

from utils.logger import Logger
from utils.dataset import CharacterDataset
from utils.sampler import BalancedBatchSampler
from utils.function import plot_sample
from model.generator import SynthesisGenerator
from model.discriminator import MultiscaleDiscriminator
from model.discriminator import StyleTemplateDiscriminator
from utils.checkpoint import CheckpointManager


@hydra.main(version_base=None, config_path='./config', config_name='training')
def main(config):
    # load configuration
    dataset_path = str(config.parameter.dataset_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    batch_size = int(config.parameter.batch_size)
    num_workers = int(config.parameter.num_workers)
    reference_count = int(config.parameter.reference_count)
    num_iterations = int(config.parameter.num_iterations)
    report_interval = int(config.parameter.report_interval)
    save_interval = int(config.parameter.save_interval)
    g_steps = int(config.parameter.schedule.g_steps)
    d_steps = int(config.parameter.schedule.d_steps)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'training.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

     # load dataset
    
    dataset = CharacterDataset(dataset_path, reference_count=reference_count) 
    batch_sampler = BalancedBatchSampler(dataset,
                                          samples_per_writer=2, 
                                          writers_per_batch=batch_size//2)
    dataloader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            num_workers=num_workers,
                            pin_memory=True)
    print('image number: {}\n'.format(len(dataset)))
    

    # create model
    generator_model = SynthesisGenerator(reference_count=reference_count).to(device)
    generator_model.train()

    discriminator_model = MultiscaleDiscriminator(dataset.writer_count, dataset.character_count).to(device)
    discriminator_model.train()

    template_D_model = StyleTemplateDiscriminator(style_dim=512, n_heads=4).to(device)
    template_D_model.train()

    # create optimizer
    generator_optimizer = Adam(generator_model.parameters(), lr=config.parameter.generator.learning_rate, betas=(0.0, 0.999), weight_decay=1e-4)
    discriminator_optimizer = Adam(discriminator_model.parameters(), lr=config.parameter.discriminator.learning_rate, betas=(0.0, 0.999), weight_decay=1e-4)
    template_D_optimizer = Adam(template_D_model.parameters(), lr=config.parameter.template_D.learning_rate, betas=(0.0, 0.999), weight_decay=1e-4)

    # start training
    current_iteration = 0
    current_time = time.time()

    # load checkpoint
    ckpt_mgr = CheckpointManager()
    if config.parameter.resume_checkpoint is not None:
        model_dict = {
            "gen": generator_model,
            "disc": discriminator_model,
            "tD":   template_D_model}
        optim_dict = {
            "gen": generator_optimizer,
            "disc": discriminator_optimizer}
        current_iteration = ckpt_mgr.load(
            config.parameter.resume_checkpoint,
            model_dict, optim_dict, 
            map_location=device)

   
            


    while current_iteration < num_iterations:
        for reference_image, writer_label, template_image, character_label, script_image in dataloader:
            current_iteration += 1
            
            for _ in range(g_steps):
                reference_image, writer_label, template_image, character_label, script_image = reference_image.to(device), writer_label.to(device), template_image.to(device), character_label.to(device), script_image.to(device)

                # generator
                generator_optimizer.zero_grad()

                result_image, template_structure, reference_style = generator_model(reference_image, template_image)



                loss_generator_adversarial = 0
                loss_generator_classification = 0
                for prediction_reality, prediction_writer, prediction_character in discriminator_model(result_image):
                    loss_generator_adversarial += F.binary_cross_entropy(prediction_reality, torch.ones_like(prediction_reality))
                    loss_generator_classification += F.cross_entropy(prediction_writer, writer_label) + F.cross_entropy(prediction_character, character_label)

                result_structure = generator_model.structure(result_image)
                loss_generator_structure = 0
                for i in range(len(result_structure)):
                    loss_generator_structure += 0.5 * torch.mean(torch.square(template_structure[i] - result_structure[i]))



                result_style = generator_model.style(result_image.repeat_interleave(reference_count, dim=1))
                l2_style = 0.5 * torch.mean(torch.square(reference_style - result_style))

                #----------------------------------------------Sup-Con-------------------------------------------------------------------
                eps = 1e-6                                                       #除零保護
                style_norm = result_style / (result_style.norm(dim=1, keepdim=True) + eps)

                tau = config.parameter.supcon.temperature
                logits = torch.div(style_norm @ style_norm.T, tau).clamp(-50, 50)
    
                 # mask[i,j] =1 if 同 writer else 0
                mask = writer_label.unsqueeze(1).eq(writer_label).float()
    
                 # 避免把自己當作正樣本
                logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
                exp_logits = torch.exp(logits) * logits_mask
    
                 # Sup-Con loss 本體
                supcon_numerator   = (exp_logits * mask).sum(dim=1)
                supcon_denominator = exp_logits.sum(dim=1)
                loss_supcon = -torch.log(supcon_numerator / supcon_denominator + 1e-8).mean()
    
                 # 合併兩種 style loss
                lambda_supcon = config.parameter.supcon.weight
                loss_generator_style = l2_style + lambda_supcon * loss_supcon
                #------------------------------------------------------------------------------------------------------------------------
                #---------------------------------- template_D -------------------------------------
                loss_generator_template_D = torch.tensor(0.0).to(device)
                if current_iteration > config.parameter.template_D.warmup_iter:
                    logits_fake = template_D_model(result_image, template_image, reference_style.detach())
                    loss_generator_template_D += F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
                #-----------------------------------------------------------------------------------

                loss_generator_reconstruction = F.l1_loss(result_image, script_image)

                loss_generator = config.parameter.generator.loss_function.weight_adversarial * loss_generator_adversarial \
                    + config.parameter.generator.loss_function.weight_classification * loss_generator_classification \
                    + config.parameter.generator.loss_function.weight_structure * loss_generator_structure \
                    + config.parameter.generator.loss_function.weight_style * loss_generator_style \
                    + config.parameter.generator.loss_function.weight_reconstruction * loss_generator_reconstruction \
                    + config.parameter.template_D.weight * loss_generator_template_D
                loss_generator.backward()
                torch.nn.utils.clip_grad_norm_(generator_model.parameters(), 1.0)   
                generator_optimizer.step()


            # discriminator
            for _ in range(d_steps):
                
                discriminator_optimizer.zero_grad()

                loss_discriminator_adversarial = 0
                loss_discriminator_classification = 0
                for prediction_reality, prediction_writer, prediction_character in discriminator_model(result_image.detach()):
                    loss_discriminator_adversarial += F.binary_cross_entropy(prediction_reality, torch.zeros_like(prediction_reality))
                    loss_discriminator_classification += F.cross_entropy(prediction_writer, writer_label) + F.cross_entropy(prediction_character, character_label)

                for prediction_reality, prediction_writer, prediction_character in discriminator_model(script_image):
                    loss_discriminator_adversarial += F.binary_cross_entropy(prediction_reality, torch.ones_like(prediction_reality))
                    loss_discriminator_classification += F.cross_entropy(prediction_writer, writer_label) + F.cross_entropy(prediction_character, character_label)

                loss_discriminator = config.parameter.discriminator.loss_function.weight_adversarial * loss_discriminator_adversarial + config.parameter.discriminator.loss_function.weight_classification * loss_discriminator_classification
                loss_discriminator.backward()
                torch.nn.utils.clip_grad_norm_(discriminator_model.parameters(), 1.0) ### <-- 必改
                discriminator_optimizer.step()
                #---------------------------------- template_D -------------------------------------
                loss_discriminator_template_D = 0
                if current_iteration > config.parameter.template_D.warmup_iter:
                    template_D_optimizer.zero_grad()
                    #fake
                    logits_fake = template_D_model(result_image.detach(), template_image, reference_style.detach())
                    loss_discriminator_template_D += F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
                    #real
                    logits_real = template_D_model(script_image, template_image, reference_style.detach())
                    loss_discriminator_template_D += F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))


                    loss_discriminator_template_D.backward()
                    torch.nn.utils.clip_grad_norm_(template_D_model.parameters(), 1.0)
                    template_D_optimizer.step()
                else:
                    loss_discriminator_template_D = torch.tensor(0.0).to(device)
                #-----------------------------------------------------------------------------------    

            # report
            if current_iteration % report_interval == 0:
                last_time = current_time
                current_time = time.time()
                iteration_time = (current_time - last_time) / report_interval

                print('iteration {} / {}:'.format(current_iteration, num_iterations))
                print('time: {:.6f} seconds per iteration'.format(iteration_time))
                print('generator loss: {:.6f}, adversarial loss: {:.6f}, classification loss: {:.6f}, structure loss: {:.6f}, l2_style loss: {:.6f}, Sup-Con: {:.6f}, reconstruction loss: {:.6f}, template_D loss: {:.6f}'\
                      .format(loss_generator.item(), loss_generator_adversarial.item(), loss_generator_classification.item(), loss_generator_structure.item(), l2_style.item(), loss_supcon.item(), loss_generator_reconstruction.item(), loss_generator_template_D.item()))
                print('discriminator loss: {:.6f}, adversarial loss: {:.6f}, classification loss: {:.6f}'.format(loss_discriminator.item(), loss_discriminator_adversarial.item(), loss_discriminator_classification.item()))
                print('template_D loss: {:.6f}\n'.format(loss_discriminator_template_D.item()))

            # save
            if current_iteration % save_interval == 0:
                save_path = os.path.join(checkpoint_path, 'iteration_{}'.format(current_iteration))
                os.makedirs(save_path, exist_ok=True)

                model_path = os.path.join(save_path, 'model.pth')
                ckpt_mgr.save(iteration=current_iteration,
                           models={"gen": generator_model,
                                   "disc": discriminator_model,
                                   "tD":   template_D_model},
                           optimizers={"gen": generator_optimizer,
                                       "disc": discriminator_optimizer,
                                       "tD":   template_D_optimizer},
                           path=model_path)
                
                image_path = os.path.join(save_path, 'sample.png')
                image = plot_sample(reference_image, template_image, script_image, result_image)[0]
                Image.fromarray((255 * image).astype(np.uint8)).save(image_path)
                print('save sample image in: {}'.format(image_path))
                
            if current_iteration >= num_iterations:
                break


if __name__ == '__main__':
    main()