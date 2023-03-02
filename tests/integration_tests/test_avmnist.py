# Copyright (c) 2022 Predibase, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import time

import models.auxiliary.scheduler as sc
import models.search.train_searchable.avmnist as tr
import numpy as np
import torch
import torch.optim as op

from ludwig.datasets.loaders.av_mnist import AV_MNISTLoader

# %% Parse inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Modality optimization.")
    parser.add_argument(
        "--checkpointdir", type=str, help="output base dir", default="/workspace/mfas/Checkpoints/AVMNIST/"
    )
    parser.add_argument("--datadir", type=str, help="data directory", default="/workspace/mfas/datasets/avmnist/")
    parser.add_argument(
        "--audio_cp",
        type=str,
        help="Audio net checkpoint (assuming is contained in checkpointdir)",
        default="audio_model.pkl",
    )
    parser.add_argument(
        "--rgb_cp",
        type=str,
        help="RGB net checkpoint (assuming is contained in checkpointdir)",
        default="image_model.pkl",
    )
    parser.add_argument(
        "--test_cp", type=str, help="Full net checkpoint (assuming is contained in checkpointdir)", default=""
    )
    parser.add_argument("--num_outputs", type=int, help="output dimension", default=10)
    parser.add_argument("--batchsize", type=int, help="batch size", default=32)
    parser.add_argument(
        "--inner_representation_size", type=int, help="output size of mixing linear layers", default=256
    )
    parser.add_argument("--epochs", type=int, help="training epochs", default=70)
    parser.add_argument("--eta_max", type=float, help="eta max", default=0.001)
    parser.add_argument("--eta_min", type=float, help="eta min", default=0.000001)
    parser.add_argument(
        "--use_dataparallel", help="Use several GPUs", action="store_true", dest="use_dataparallel", default=True
    )
    parser.add_argument("--j", dest="num_workers", type=int, help="Dataloader CPUS", default=12)
    parser.add_argument("--modality", type=str, help="", default="both")
    parser.add_argument("--no-verbose", help="verbose", action="store_false", dest="verbose", default=True)
    parser.add_argument("--weightsharing", help="Weight sharing", action="store_true", default=False)
    parser.add_argument("--multitask", dest="multitask", help="Multitask loss", action="store_true", default=False)
    parser.add_argument("--alphas", help="Use alphas", action="store_true", default=False)
    parser.add_argument("--batchnorm", help="Use batch norm", action="store_true", dest="batchnorm", default=False)
    parser.add_argument("--channels", type=int, default=3)

    parser.add_argument("--Ti", type=int, help="epochs Ti", default=5)
    parser.add_argument("--Tm", type=int, help="epochs multiplier Tm", default=2)
    parser.add_argument("--drpt", action="store", default=0.4, dest="drpt", type=float, help="dropout")

    parser.add_argument("--conf", type=int, help="conf to train", default=0)

    return parser.parse_args()


# %%
def get_dataloaders(args):
    import torchvision.transforms as transforms
    from datasets import avmnist as d
    from torch.utils.data import DataLoader

    # Handle data
    transformer_val = transforms.Compose([d.ToTensor(), d.Normalize((0.1307,), (0.3081,))])
    transformer_tra = transforms.Compose([d.ToTensor(), d.Normalize((0.1307,), (0.3081,))])

    dataset_training = d.AVMnist(args.datadir, transform=transformer_tra, stage="train")
    dataset_testing = d.AVMnist(args.datadir, transform=transformer_val, stage="test")
    dataset_validation = d.AVMnist(args.datadir, transform=transformer_val, stage="dev")

    datasets = {"train": dataset_training, "dev": dataset_validation, "test": dataset_testing}

    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        for x in ["train", "dev", "test"]
    }

    return dataloaders


def train_model(rmode, configuration, dataloaders, args, device):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "test", "dev"]}

    if args.test_cp == "":
        num_batches_per_epoch = dataset_sizes["train"] / args.batchsize
        criteria = [torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()]

        # loading pretrained weights
        audmodel_filename = os.path.join(args.checkpointdir, args.audio_cp)
        rgbmodel_filename = os.path.join(args.checkpointdir, args.rgb_cp)
        sd_a = torch.load(audmodel_filename)
        sd_r = torch.load(rgbmodel_filename)

        for key in list(sd_a.keys()):
            if "module." in key:
                sd_a[key.replace("module.", "")] = sd_a.pop(key)

        for key in list(sd_r.keys()):
            if "module." in key:
                sd_r[key.replace("module.", "")] = sd_r.pop(key)

        rmode.audnet.load_state_dict(sd_a)
        rmode.rgbnet.load_state_dict(sd_r)

        # optimizer and scheduler
        params = rmode.central_params()
        optimizer = op.Adam(params, lr=args.eta_max / 10, weight_decay=1e-4)
        scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm, num_batches_per_epoch)

        # hardware tuning
        if torch.cuda.device_count() > 1 and args.use_dataparallel:
            rmode = torch.nn.DataParallel(rmode)
        rmode.to(device)

        if args.verbose:
            print("Pretraining central weights: ")
            print(configuration)

        interm_model_acc = tr.train_avmnist_track_acc(
            rmode,
            criteria,
            optimizer,
            scheduler,
            dataloaders,
            dataset_sizes,
            device=device,
            num_epochs=1,
            verbose=args.verbose,
            multitask=args.multitask,
        )

        if args.verbose:
            print("Intermediate val accuracy: " + str(interm_model_acc))

        if torch.cuda.device_count() > 1 and args.use_dataparallel:
            params = rmode.module.parameters()
        else:
            params = rmode.parameters()

        optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)
        scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm, num_batches_per_epoch)
        best_model_acc = tr.train_avmnist_track_acc(
            rmode,
            criteria,
            optimizer,
            scheduler,
            dataloaders,
            dataset_sizes,
            device=device,
            num_epochs=args.epochs,
            verbose=args.verbose,
            multitask=args.multitask,
        )

        if args.verbose:
            print("Final val accuracy: " + str(best_model_acc))

    else:
        # perform test only (load weights then)
        fullmodel_filename = os.path.join(args.checkpointdir, args.test_cp)
        rmode.load_state_dict(torch.load(fullmodel_filename))

        # hardware tuning
        if torch.cuda.device_count() > 1 and args.use_dataparallel:
            rmode = torch.nn.DataParallel(rmode)
        rmode.to(device)

    test_model_acc = tr.test_avmnist_track_acc(
        rmode, dataloaders, dataset_sizes, device=device, multitask=args.multitask
    )

    if args.verbose:
        print("Final test accuracy: " + str(test_model_acc))

    return test_model_acc


# %%
if __name__ == "__main__":
    # %%
    print("Training found AVMNIST network")
    args = parse_args()
    print("The configuration of this run is:")
    print(args)

    # %% hardware
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # %% Train found
    # Now listing best architectures
    #       [(array([[3, 2, 1],
    #        [3, 2, 1],
    #        [2, 0, 1],
    #        [2, 2, 0]]), tensor(0.9430, device='cuda:0')),
    #
    #        (array([[3, 2, 0]]), tensor(0.9434, device='cuda:0')),
    #
    #        (array([[1, 1, 1],
    #        [2, 0, 0],
    #        [0, 2, 0],
    #        [2, 2, 0]]), tensor(0.9436, device='cuda:0')),
    #
    #        (array([[1, 1, 1],
    #        [0, 2, 0],
    #        [3, 2, 1],
    #        [2, 2, 0]]), tensor(0.9488, device='cuda:0')),
    #
    #        (array([[2, 2, 0],
    #        [3, 1, 1],
    #        [0, 2, 0],
    #        [2, 2, 0]]), tensor(0.9500, device='cuda:0'))]
    if args.conf == 0:
        configuration = np.array([[3, 2, 1], [3, 2, 1], [2, 0, 1], [2, 2, 0]])
    elif args.conf == 1:
        configuration = np.array([[3, 2, 0]])
    elif args.conf == 2:
        configuration = np.array([[1, 1, 1], [2, 0, 0], [0, 2, 0], [2, 2, 0]])
    elif args.conf == 3:
        configuration = np.array([[1, 1, 1], [0, 2, 0], [3, 2, 1], [2, 2, 0]])
    elif args.conf == 4:
        configuration = np.array([[2, 2, 0], [3, 1, 1], [0, 2, 0], [2, 2, 0]])

    rmode = avmnist.Searchable_Audio_Image_Net(args, configuration)
    dataloaders = get_dataloaders(args)
    start_time = time.time()
    modelacc = train_model(rmode, configuration, dataloaders, args, device)
    time_elapsed = time.time() - start_time
    print(f"Training in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Model Acc: {modelacc}")

    # %%
    confstr = np.array2string(configuration, precision=1, separator="_", suppress_small=True)
    confstr = re.sub(r"_\n ", "_", confstr)

    # filename = args.checkpointdir+"/final_conf_" + confstr + "_" + str(modelacc.item())+'.checkpoint'
    # torch.save(rmode.state_dict(), filename)
