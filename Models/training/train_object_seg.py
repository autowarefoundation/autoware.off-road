#! /usr/bin/env python3

import torch
import random
from argparse import ArgumentParser
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
    args = parser.parse_args()

    # Root path
    root = args.root

    # Model save path
    model_save_root_path = args.model_save_root_path

    # Load from checkpoint
    load_from_checkpoint = False
    if(args.load_from_save):
        load_from_checkpoint = True

    # Data paths
    # CASSED
    cassed_labels_filepath = root + 'CASSED/gt_masks/'
    cassed_images_filepath = root + 'CASSED/images/'

    # GOOSE
    goose_labels_fileapath = root + 'GOOSE/gt_masks/'
    goose_images_fileapath = root + 'GOOSE/images/'

    # OFFSED
    offsed_labels_fileapath = root + 'OFFSED/gt_masks/'
    offsed_images_fileapath = root + 'OFFSED/images/'

    # RELLIS3D
    rellis3d_labels_fileapath = root + 'RELLIS3D/gt_masks/'
    rellis3d_images_fileapath = root + 'RELLIS3D/images/'

    # CASSED - Data Loading
    cassed_Dataset = LoadDataObjectSeg(
        cassed_labels_filepath, cassed_images_filepath, 'CASSED')
    cassed_num_train_samples, cassed_num_val_samples = cassed_Dataset.getItemCount()

    # GOOSE - Data Loading
    goose_Dataset = LoadDataObjectSeg(
        goose_labels_fileapath, goose_images_fileapath, 'GOOSE')
    goose_num_train_samples, goose_num_val_samples = goose_Dataset.getItemCount()

    # OFFSED - Data Loading
    offsed_Dataset = LoadDataObjectSeg(
        offsed_labels_fileapath, offsed_images_fileapath, 'OFFSED')
    offsed_num_train_samples, offsed_num_val_samples = offsed_Dataset.getItemCount()

    # RELLIS3D - Data Loading
    rellis3d_Dataset = LoadDataObjectSeg(
        rellis3d_labels_fileapath, rellis3d_images_fileapath, 'RELLIS3D')
    rellis3d_num_train_samples, rellis3d_num_val_samples = rellis3d_Dataset.getItemCount()

    # Total number of training samples
    total_train_samples = cassed_num_train_samples + \
        goose_num_train_samples + offsed_num_train_samples \
        + rellis3d_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = cassed_num_val_samples + \
        goose_num_val_samples + offsed_num_val_samples \
        + rellis3d_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Pre-trained model checkpoint path
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    checkpoint_path = args.checkpoint_path

    # Trainer Class
    if load_from_checkpoint == False:
        trainer = ObjectSegTrainer(pretrained_checkpoint_path=pretrained_checkpoint_path)
    else:
        trainer = ObjectSegTrainer(checkpoint_path=checkpoint_path, is_pretrained=True)

    trainer.zero_grad()

    # Total training epochs
    num_epochs = 10
    batch_size = 32

    # Epochs
    for epoch in range(0, num_epochs):

        # Iterators for datasets
        cassed_count = 0
        goose_count = 0
        offsed_count = 0
        rellis3d_count = 0

        is_cassed_complete = False
        is_goose_complete = False
        is_offsed_complete = False
        is_rellis3d_complete = False

        data_list = []
        data_list.append('CASSED')
        data_list.append('GOOSE')
        data_list.append('OFFSED')
        data_list.append('RELLIS3D')
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

        # Loop through data
        for count in range(0, total_train_samples):

            log_count = count + total_train_samples*epoch

            # Reset iterators
            if cassed_count == cassed_num_train_samples and not is_cassed_complete:
                is_cassed_complete = True
                data_list.remove("CASSED")

            if goose_count == goose_num_train_samples and not is_goose_complete:
                is_goose_complete = True
                data_list.remove("GOOSE")

            if offsed_count == offsed_num_train_samples and not is_offsed_complete:
                is_offsed_complete = True
                data_list.remove('OFFSED')

            if rellis3d_count == rellis3d_num_train_samples and not is_rellis3d_complete:
                is_rellis3d_complete = True
                data_list.remove('RELLIS3D')

            if data_list_count >= len(data_list):
                data_list_count = 0

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            if data_list[data_list_count] == 'CASSED' and not is_cassed_complete:
                image, gt, class_weights = \
                    cassed_Dataset.getItemTrain(cassed_count)
                cassed_count += 1

            if data_list[data_list_count] == 'GOOSE' and not is_goose_complete:
                image, gt, class_weights = \
                    goose_Dataset.getItemTrain(goose_count)
                goose_count += 1

            if data_list[data_list_count] == 'OFFSED' and not is_offsed_complete:
                image, gt, class_weights = \
                    offsed_Dataset.getItemTrain(offsed_count)
                offsed_count += 1

            if data_list[data_list_count] == 'RELLIS3D' and not is_rellis3d_complete:
                is_rellis3d_complete = True
                image, gt, class_weights = \
                    rellis3d_Dataset.getItemTrain(rellis3d_count)
                rellis3d_count += 1

            # Assign Data
            trainer.set_data(image, gt, class_weights)

            # Augmenting Image
            trainer.apply_augmentations(is_train=True)

            # Converting to tensor and loading
            trainer.load_data(is_train=True)

            # Run model and calculate loss
            trainer.run_model()

            # Gradient accumulation
            trainer.loss_backward()

            # Simulating batch size through gradient accumulation
            if (count+1) % batch_size == 0:
                trainer.run_optimizer()

            # Logging loss to Tensor Board every 250 steps
            if (count+1) % 250 == 0:
                trainer.log_loss(log_count)

            # Logging Image to Tensor Board every 1000 steps
            if (count+1) % 1000 == 0:
                trainer.save_visualization(log_count)

            # Save model and run validation on entire validation
            # dataset after 8000 steps
            if (count+1) % 8000 == 0:

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

                    # CASSED
                    for val_count in range(cassed_num_val_samples):
                        image_val, gt_val, _ = \
                            cassed_Dataset.getItemVal(val_count)

                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # GOOSE
                    for val_count in range(goose_num_val_samples):
                        image_val, gt_val, _ = \
                            goose_Dataset.getItemVal(val_count)

                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # OFFSED
                    for val_count in range(offsed_num_val_samples):
                        image_val, gt_val, _ = \
                            offsed_Dataset.getItemVal(val_count)

                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # RELLIS3D
                    for val_count in range(rellis3d_num_val_samples):
                        image_val, gt_val, _ = \
                            rellis3d_Dataset.getItemVal(val_count)

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
