import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor


dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

print(dataset)
print("データ数:", len(dataset))

image, label = dataset[0]

print("画像:", image)
print("画像shape:", image.shape)
print("画像dtype:", image.dtype)
print("正解ラベル:", label)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.show()


from torch.utils.data import DataLoader


dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

images, labels = next(iter(dataloader))

print("images shape:", images.shape)
print("labels shape:", labels.shape)

print("labels:")
print(labels)