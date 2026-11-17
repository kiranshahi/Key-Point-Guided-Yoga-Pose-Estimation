from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class YogaDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        with Image.open(record["path"]) as img:
            image = transform(img.convert("RGB"))

        keypoints = torch.tensor(record["keypoints"], dtype=torch.float32)
        mask = torch.tensor(record["mask"], dtype=torch.float32)
        label = torch.tensor(record["label"], dtype=torch.long)

        return image, keypoints, mask, label

labels = [r["label"] for r in train_records]

train_idx, val_idx = train_test_split(np.arange(len(train_records)), test_size=0.2, stratify=labels, random_state=42)

train_data = YogaDataset([train_records[i] for i in train_idx])
val_data = YogaDataset([train_records[i] for i in val_idx])
test_data = YogaDataset(test_records)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)