from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open(r"data\train\ants_image\1030023514_aad5c608f9.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

# Normalize
trans_norm = transforms.Normalize([1, 1, 1], [1, 1, 1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

writer.close()