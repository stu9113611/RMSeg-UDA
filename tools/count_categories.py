import argparse
import json
import os.path as osp

import torch
from engine.category import Category, count_categories
from engine.dataloader import ImgAnnDataset
from engine.logger import Logger
from engine.misc import dict_add
from engine.transform import LoadAnn, Resize
from rich.progress import track


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count pixels per category in a dataset."
    )
    parser.add_argument("csv_path", type=str)
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--rcs-file-savepath", type=str, default=None)
    args = parser.parse_args()
    return args


def main(args):
    categories = Category.load(args.csv_path)

    dataloader = ImgAnnDataset(
        args.dataset_root,
        img_prefix="images",
        ann_prefix="labels",
        img_suffix=".jpg",
        ann_suffix=".png",
        transforms=[
            LoadAnn(categories),
            Resize(None),
        ],
        max_len=args.max_length,
        check_exist=False,
    ).get_loader(1, False, 0, False, False)

    if args.rcs_file_savepath:
        rcs = []

    counts = torch.zeros(len(categories))
    for data in track(dataloader, description="Counting categories..."):
        label = data["ann"]
        count = count_categories(label, categories)
        counts += count

        if args.rcs_file_savepath:
            rcs.append({"filename": data["ann_path"][0], "count": count.tolist()})

    if args.rcs_file_savepath:
        with open(args.rcs_file_savepath, "w") as f:
            json.dump(rcs, f)

    print(counts)


if __name__ == "__main__":
    args = parse_args()
    main(args)
