import os
import numpy as np
from PIL import Image
import random
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


def simple_conversion(seq):
    """Create 26-dim embedding"""
    chars = [
        "-",
        "M",
        "R",
        "H",
        "K",
        "D",
        "E",
        "S",
        "T",
        "N",
        "Q",
        "C",
        "U",
        "G",
        "P",
        "A",
        "V",
        "I",
        "F",
        "Y",
        "W",
        "L",
        "O",
        "X",
        "Z",
        "B",
        "J",
    ]

    nums = range(len(chars))

    seqs_x = np.zeros(len(seq))

    for idx, char in enumerate(seq):

        lui = chars.index(char)

        seqs_x[idx] = nums[lui]

    return torch.tensor([seqs_x]).long()


def convert_descriptor(seq):
    seq_dict = {
        "<pad>": 0,
        "M": 1,
        "R": 2,
        "H": 3,
        "K": 4,
        "D": 5,
        "E": 6,
        "S": 7,
        "T": 8,
        "N": 9,
        "Q": 10,
        "C": 11,
        "G": 12,
        "P": 13,
        "A": 14,
        "V": 15,
        "I": 16,
        "F": 17,
        "Y": 18,
        "W": 19,
        "L": 20,
        "<cls>": 21,
    }
    seq = seq.upper()
    return torch.tensor([seq_dict[char] for char in seq]).long()


class OpenCellLoader(Dataset):
    """imports mined opencell images with protein sequence"""

    def __init__(
        self,
        config_file,
        split_key=None,
        crop_size=600,
        crop_method="random",
        sequence_mode="simple",
        vocab="bert",
        threshold=False,
        text_seq_len=0,
    ):
        self.config_file = config_file
        self.image_folders = []
        self.crop_method = crop_method
        self.crop_size = crop_size
        self.sequence_mode = sequence_mode
        self.threshold = threshold
        self.text_seq_len = int(text_seq_len)
        self.vocab = vocab

        if self.sequence_mode == "embedding" or self.sequence_mode == "onehot":

            from tape import TAPETokenizer

            if self.vocab == "unirep" or self.sequence_mode == "onehot":
                self.tokenizer = TAPETokenizer(vocab="unirep")

            elif self.vocab == "bert":
                self.tokenizer = TAPETokenizer(vocab="iupac")

            elif self.vocab == "esm1b":
                from esm import Alphabet

                self.tokenizer = Alphabet.from_architecture(
                    "ESM-1b"
                ).get_batch_converter()

        data = pd.read_csv(config_file)

        self.parent_path = os.path.dirname(config_file).split(config_file)[0]

        if split_key == "train":
            self.data = data[data["split"] == "train"]
        elif split_key == "val":
            self.data = data[data["split"] == "val"]
        else:
            self.data = data

        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_vector = self.get_protein_vector(idx)

        nucleus, target, threshold = self.get_images(idx)

        data_dict = {
            "nucleus": nucleus.float(),
            "target": target.float(),
            "threshold": threshold.float(),
            "sequence": protein_vector.long(),
        }

        return data_dict

    def get_protein_vector(self, idx):

        if "protein_sequence" not in self.data.columns:

            metadata = self.retrieve_metadata(idx)
            protein_sequence = metadata["sequence"]
        else:
            protein_sequence = self.data.iloc[idx]["protein_sequence"]

        protein_vector = self.tokenize_seqeuence(protein_sequence)

        return protein_vector

    def get_images(self, idx):

        nucleus = Image.open(
            os.path.join(self.parent_path, self.data.iloc[idx]["nucleus_image_path"])
        )
        target = Image.open(
            os.path.join(self.parent_path, self.data.iloc[idx]["protein_image_path"])
        )

        # from https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914

        if self.crop_method == "random":

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                nucleus, output_size=(self.crop_size, self.crop_size)
            )

            nucleus = TF.crop(nucleus, i, j, h, w)
            target = TF.crop(target, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                nucleus = TF.hflip(nucleus)
                target = TF.hflip(target)

            # Random vertical flipping
            if random.random() > 0.5:
                nucleus = TF.vflip(nucleus)
                target = TF.vflip(target)

        elif self.crop_method == "center":
            nucleus = TF.center_crop(nucleus, self.crop_size)
            target = TF.center_crop(target, self.crop_size)

        nucleus = TF.to_tensor(nucleus)
        target = TF.to_tensor(target)

        threshold = target

        if self.threshold:
            threshold = 1.0 * (threshold > (torch.mean(threshold)))

        return nucleus, target, threshold

    def retrieve_metadata(self, idx):

        with open(
            os.path.join(self.parent_path, self.data.iloc[idx]["metadata_path"])
        ) as f:
            metadata = json.load(f)

        return metadata

    def tokenize_seqeuence(self, protein_sequence):

        prot_len = len(protein_sequence)

        if prot_len > self.text_seq_len:
            start_int = np.random.randint(0, len(protein_sequence) - self.text_seq_len)
            protein_sequence = protein_sequence[
                start_int : start_int + self.text_seq_len
            ]

        if self.sequence_mode == "simple":
            protein_vector = simple_conversion(protein_sequence)

        elif self.sequence_mode == "center":
            protein_sequence = protein_sequence.center(self.text_seq_length, "-")
            protein_vector = simple_conversion(protein_sequence)

        elif self.sequence_mode == "alternating":
            protein_sequence = protein_sequence.center(self.text_seq_length, "-")
            protein_sequence = protein_sequence[::18]
            protein_sequence = protein_sequence.center(
                int(self.text_seq_length / 18) + 1, "-"
            )
            protein_vector = simple_conversion(protein_sequence)

        elif self.sequence_mode == "onehot":

            protein_vector = torch.tensor([self.tokenizer.encode(protein_sequence)])[
                :, 1:-1
            ]

        elif self.sequence_mode == "aadescriptors":

            protein_vector = convert_descriptor(protein_sequence).long().unsqueeze(0)

        elif self.sequence_mode == "embedding":

            if self.vocab == "esm1b":
                pad_token = 3

                protein_vector = self.tokenizer([("", protein_sequence)])[-1][:, 1:]

            elif self.vocab == "unirep" or self.vocab == "bert":
                pad_token = 0
                protein_vector = torch.tensor(
                    [self.tokenizer.encode(protein_sequence)]
                )[:, 1:]


            if prot_len > self.text_seq_len:
                protein_vector = protein_vector[:, :-1]
            elif prot_len == self.text_seq_len:
                protein_vector = protein_vector[:, :-2]

            if protein_vector.shape[-1] < self.text_seq_len:
                diff = self.text_seq_len - protein_vector.shape[-1]
                protein_vector = torch.nn.functional.pad(
                    protein_vector, (0, diff), "constant", pad_token
                )

            return protein_vector.long()
        
        else:
            
            assert("No valid sequence mode selected")
        
        if protein_vector.shape[-1] + 1 < self.text_seq_len:
            diff = self.text_seq_len - protein_vector.shape[-1]
            protein_vector = torch.nn.functional.pad(
                protein_vector, (0, diff), "constant", 0
            )

        return protein_vector.long()
