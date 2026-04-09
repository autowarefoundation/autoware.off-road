#! /usr/bin/env python3

import os
import sys
import numpy as np
import random
from tqdm import tqdm
from argparse import ArgumentParser
sys.path.append('..')
from data_utils.load_data_elevation import LoadDataElevation, DATASETS
from training.elevation_trainer import ElevationTrainer


def main():

    parser = ArgumentParser()
    parser.add_argument('-s', '--model_save_root_path', dest='model_save_root_path', required=True,
                        help='root path where PyTorch checkpoint files will be saved')
    parser.add_argument('-c', '--checkpoint_path', dest='checkpoint_path', default='',
                        help='path to an existing ElevationNetwork checkpoint to resume training')
    parser.add_argument('-r', '--root', dest='root', required=True,
                        help='root path to the AORAS/Elevation/ data directory')
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('-a', '--start_epoch', dest='start_epoch', default=0, type=int,
                        help='starting epoch index (for resuming training)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.0001, type=float,
                        help='AdamW learning rate')
    args = parser.parse_args()

    os.makedirs(args.model_save_root_path, exist_ok=True)

    # ── Load datasets ─────────────────────────────────────────────────────────
    datasets_info = {
        name: {
            'labels':        os.path.join(args.root, name, name, 'gt_masks/'),
            'images':        os.path.join(args.root, name, name, 'images/'),
            'camera_params': os.path.join(args.root, name, name, 'camera_params/'),
        }
        for name in DATASETS
    }

    datasets = {
        name: LoadDataElevation(
            labels_filepath=datasets_info[name]['labels'],
            images_filepath=datasets_info[name]['images'],
            camera_params_filepath=datasets_info[name]['camera_params'],
            dataset=name,
        )
        for name in DATASETS
    }

    counts = {name: datasets[name].getItemCount() for name in DATASETS}
    total_train_samples = sum(v[0] for v in counts.values())
    total_val_samples   = sum(v[1] for v in counts.values())
    print(total_train_samples, ': total training samples')
    print(total_val_samples,   ': total validation samples')

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = ElevationTrainer(
        checkpoint_path=args.checkpoint_path,
        learning_rate=args.learning_rate,
    )
    trainer.zero_grad()

    num_epochs  = args.num_epochs
    start_epoch = args.start_epoch

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + num_epochs):

        batch_size = _batch_size_schedule(epoch)
        losses     = []

        train_counts = {name: 0     for name in DATASETS}
        completed    = {name: False for name in DATASETS}

        data_list       = DATASETS.copy()
        random.shuffle(data_list)
        data_list_count = 0

        for count in tqdm(range(total_train_samples),
                          desc=f'Training Epoch {epoch + 1} / {start_epoch + num_epochs}'):

            log_count = count + total_train_samples * epoch

            # Retire exhausted datasets from this epoch's queue
            for name in list(data_list):
                if train_counts[name] == counts[name][0] and not completed[name]:
                    completed[name] = True
                    data_list.remove(name)

            if data_list_count >= len(data_list):
                data_list_count = 0

            dataset_name = data_list[data_list_count]
            image, label, camera_params = datasets[dataset_name].getItemTrain(
                train_counts[dataset_name])
            train_counts[dataset_name] += 1

            trainer.set_data(image, label, camera_params)
            trainer.apply_augmentations(is_train=True)
            trainer.load_data(is_train=True)
            trainer.run_model()
            losses.append(trainer.get_loss())
            trainer.loss_backward()

            # Gradient accumulation
            if (count + 1) % batch_size == 0 or (count + 1) == total_train_samples:
                trainer.run_optimizer()

            # Log training loss every 250 steps
            if (count + 1) % 250 == 0 or (count + 1) == total_train_samples:
                trainer.log_loss(log_count, loss=np.mean(losses))
                losses.clear()

            # Save visualization every 1000 steps
            if (count + 1) % 1000 == 0 or (count + 1) == total_train_samples:
                trainer.save_visualization(log_count)

            # Checkpoint + full validation every 8000 steps
            if (count + 1) % 8000 == 0 or (count + 1) == total_train_samples:

                model_save_path = os.path.join(
                    args.model_save_root_path,
                    f'iter_{count + total_train_samples * epoch}'
                    f'_epoch_{epoch}_step_{count}.pth'
                )
                trainer.save_model(model_save_path)

                print('Validating')
                trainer.set_eval_mode()
                running_mae = 0.0

                with torch.no_grad():
                    for dataset_name in DATASETS:
                        for val_count in tqdm(range(counts[dataset_name][1]),
                                              desc=f'Validating {dataset_name}'):
                            image_val, label_val, cam_val = \
                                datasets[dataset_name].getItemVal(val_count)
                            running_mae += trainer.validate(image_val, label_val, cam_val)

                mean_mae = running_mae / total_val_samples
                trainer.log_MAE(mean_mae, log_count)
                trainer.set_train_mode()

            data_list_count += 1

    trainer.cleanup()


def _batch_size_schedule(epoch: int) -> int:
    if epoch < 10:  return 16
    if epoch < 30:  return 8
    if epoch < 60:  return 4
    if epoch < 80:  return 2
    return 1


if __name__ == '__main__':
    import torch   # imported here so the no-grad block above can reference it
    main()
