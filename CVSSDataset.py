from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import csv

class CVSSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_cvss_txt(split_dir, list_classes):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["LOW", "HIGH"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            for i in range(len(list_classes)):
                if list_classes[i] == label_dir:
                    labels.append(i)
                else:
                    continue

    return texts, labels

def read_cvss_csv(file_name, num_label, list_classes):
    texts      = []
    labels     = []

    csv_file   = open(file_name, 'r+')
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')

    for row in csv_reader:
        texts.append(row[0])
        for i in range(len(list_classes)):
            if list_classes[i] == row[num_label]:
                labels.append(i)
            else:
                continue

    csv_file.close()

    return texts, labels