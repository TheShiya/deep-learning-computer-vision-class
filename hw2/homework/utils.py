from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, data_limit=-1):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        with open(os.path.join(dataset_path, 'labels.csv')) as labels_file:
            self.dataset_path = dataset_path
            csv_reader = csv.reader(labels_file, delimiter=',')
            data = list(csv_reader)[1:] # Exclude headers
            data = data[:data_limit]
            self.length = len(data)

            labels_to_int = dict(zip(LABEL_NAMES, range(len(LABEL_NAMES))))
            self.labels = [labels_to_int[row[1]] for row in data]
            self.tracks = [row[2] for row in data]
            self.image_filenames = [row[0] for row in data]
            self.tensors = []
            self.to_tensor = transforms.ToTensor()


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        image_filename = self.image_filenames[idx]
        image = Image.open(os.path.join(self.dataset_path, image_filename))
        return self.to_tensor(image), self.labels[idx]


def load_data(dataset_path, num_workers=0, batch_size=128, data_limit=9999999):
    dataset = SuperTuxDataset(dataset_path, data_limit)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
