# transforms基础用法合集
# b站小土堆Python深度学习课程笔记
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

# Resize
print(img.size)
trans_resize = transforms.Resize((233,233))

# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
print(img_resize)
# img_resize PIL -> ToTensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
print(img_resize)
writer.add_image("Resize", img_resize, 0)

# Compose （另一种resize方法）
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor]) #流水线式组合操作，将多个步骤合并到一行代码执行
img_resize_2 = trans_compose(img)
writer.add_image("Resize-2", img_resize_2, 0)

writer.close()
