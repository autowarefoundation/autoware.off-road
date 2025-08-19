#! /usr/bin/env python3

import torch
import numpy as np
from tqdm import tqdm
import random
from argparse import ArgumentParser
import os
import sys
sys.path.append('..')
from data_utils.load_data_object_seg import LoadDataObjectSeg
from training.object_seg_trainer import ObjectSegTrainer


def main():

    parser = ArgumentParser()
    parser.add_argument("-s", "--model_save_root_path", dest="model_save_root_path",
                        help="root path where pytorch checkpoint file should be saved")
    parser.add_argument("-m", "--pretrained_checkpoint_path", dest="pretrained_checkpoint_path",
                        help="path to SceneSeg weights file for pre-trained backbone")
    parser.add_argument("-c", "--checkpoint_path", dest="checkpoint_path",
                        help="path to ObjectSeg weights file for training from saved checkpoint")
    parser.add_argument("-r", "--root", dest="root",
                        help="root path to folder where data training data is stored")
    parser.add_argument("-t", "--test_images_save_root_path", dest="test_images_save_root_path",
                        help="root path where test images are stored")
    parser.add_argument('-l', "--load_from_save", action='store_true',
                        help="flag for whether model is being loaded from a ObjectSeg checkpoint file")
    parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int,
                        help="number of training epochs")
    parser.add_argument("-a", "--start_epoch", dest="start_epoch", default=0, type=int,
                        help="starting epoch for training")
    parser.add_argument("--learning_rate", dest="learning_rate", default=0.0001, type=float,
                        help="learning rate for training")
    args = parser.parse_args()

    # Root path
    root = args.root

    # Model save path
    model_save_root_path = args.model_save_root_path
    os.makedirs(model_save_root_path, exist_ok=True)

    # Load from checkpoint
    load_from_checkpoint = False
    if args.load_from_save:
        load_from_checkpoint = True

    datasets_name = ["CaSSeD", "Goose", "OFFSED", "ORFD", "Rellis_3D", "Yamaha_CMU"]
    datasets_info = {name: {"labels": root + f"{name}/{name}/gt_masks/",
                            "images": root + f"{name}/{name}/images/"}
                             for name in datasets_name}
    datasets = {name: LoadDataObjectSeg(datasets_info[name]["labels"],
                                        datasets_info[name]["images"],
                                        name) for name in datasets_name}
    counts = {name: datasets[name].getItemCount() for name in datasets}

    total_train_samples = sum(v[0] for v in counts.values())
    total_val_samples = sum(v[1] for v in counts.values())
    print(total_train_samples, ": total training samples")
    print(total_val_samples, ": total validation samples")

    # Pre-trained model checkpoint path
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    checkpoint_path = args.checkpoint_path

    # Trainer Class
    if load_from_checkpoint == False:
        trainer = ObjectSegTrainer(pretrained_checkpoint_path=pretrained_checkpoint_path,
                                   learning_rate=args.learning_rate)
    else:
        trainer = ObjectSegTrainer(checkpoint_path=checkpoint_path,
                                   is_pretrained=True,
                                   learning_rate=args.learning_rate)

    trainer.zero_grad()

    # Total training epochs
    num_epochs = args.num_epochs
    start_epoch = args.start_epoch
    batch_size = 32

    # Epochs
    for epoch in range(start_epoch, start_epoch + num_epochs):

        # Iterators for datasets
        train_counts = {name: 0 for name in datasets}
        completed = {name: False for name in datasets}

        data_list = datasets_name.copy()
        random.shuffle(data_list)
        data_list_count = 0

        if epoch == 1:
            batch_size = 16
        if epoch == 2:
            batch_size = 8
        if epoch == 3:
            batch_size = 5
        if 4 <= epoch < 6:
            batch_size = 3
        if 6 <= epoch < 8:
            batch_size = 2
        if epoch > 8:
            batch_size = 1

        losses = []

        # Loop through data
        for count in tqdm(range(0, total_train_samples), desc=f"Training Epoch {epoch + 1} / {start_epoch + num_epochs}"):

            log_count = count + total_train_samples*epoch

            # Reset dataset iterators if exhausted
            for name in list(data_list):
                if train_counts[name] == counts[name][0] and not completed[name]:
                    completed[name] = True
                    data_list.remove(name)

            if data_list_count >= len(data_list):
                data_list_count = 0

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            dataset_name = data_list[data_list_count]
            image, gt, class_weights = datasets[dataset_name].getItemTrain(train_counts[dataset_name])
            train_counts[dataset_name] += 1

            # Assign Data
            trainer.set_data(image, gt, class_weights)

            # Augmenting Image
            trainer.apply_augmentations(is_train=True)

            # Converting to tensor and loading
            trainer.load_data(is_train=True)

            # Run model and calculate loss
            trainer.run_model()
            losses.append(trainer.get_loss())

            # Gradient accumulation
            trainer.loss_backward()

            # Simulating batch size through gradient accumulation
            if ((count+1) % batch_size == 0) or ((count+1) == total_train_samples):
                trainer.run_optimizer()

            # Logging loss to Tensor Board every 250 steps
            if ((count+1) % 250 == 0) or ((count+1) == total_train_samples):
                trainer.log_loss(log_count, loss=np.mean(losses))
                losses.clear()

            # Logging Image to Tensor Board every 1000 steps
            if ((count+1) % 1000 == 0) or ((count+1) == total_train_samples):
                trainer.save_visualization(log_count)

            # Save model and run validation on entire validation
            # dataset after 8000 steps
            if ((count+1) % 8000 == 0) or ((count+1) == total_train_samples):

                # Save Model
                model_save_path = model_save_root_path + 'iter_' + \
                    str(count + total_train_samples*epoch) \
                    + '_epoch_' + str(epoch) + '_step_' + \
                    str(count) + '.pth'

                trainer.save_model(model_save_path)

                # Validate
                print('Validating')

                # Setting model to evaluation mode
                trainer.set_eval_mode()

                running_IoU_full = 0
                running_IoU_bg = 0
                running_IoU_fg = 0
                running_IoU_rd = 0

                # No gradient calculation
                with torch.no_grad():

                    for dataset_name in datasets_name:
                        for val_count in tqdm(range(0, counts[dataset_name][1]),
                                              desc=f"Validating {dataset_name}"):
                            image_val, gt_val, _ = \
                                datasets[dataset_name].getItemVal(val_count)

                            # Run Validation and calculate IoU Score
                            IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                                trainer.validate(image_val, gt_val)

                            running_IoU_full += IoU_score_full
                            running_IoU_bg += IoU_score_bg
                            running_IoU_fg += IoU_score_fg
                            running_IoU_rd += IoU_score_rd

                    # Calculating average loss of complete validation set
                    mIoU_full = running_IoU_full/total_val_samples
                    mIoU_bg = running_IoU_bg/total_val_samples
                    mIoU_fg = running_IoU_fg/total_val_samples
                    mIoU_rd = running_IoU_rd/total_val_samples

                    # Logging average validation loss to TensorBoard
                    trainer.log_IoU(mIoU_full, mIoU_bg,
                                    mIoU_fg, mIoU_rd, log_count)

                # Resetting model back to training
                trainer.set_train_mode()

            data_list_count += 1

    trainer.cleanup()


if __name__ == '__main__':
    main()
