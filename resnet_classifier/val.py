import torch  
import torchvision.transforms as transforms  
from torchvision import models  
from PIL import Image  
  
# 加载训练好的模型  
model_path = 'classifier.pth'  
model = models.resnet18(pretrained=False)  # 不使用预训练的权重  
num_ftrs = model.fc.in_features  # 获取全连接层的输入特征数  
model.fc = torch.nn.Linear(num_ftrs, 2)  # 修改全连接层为二分类  
model.load_state_dict(torch.load(model_path))  # 加载模型权重  
model.eval()  # 设置为评估模式  
  
# 定义图像预处理步骤  
preprocess = transforms.Compose([  
    transforms.Resize(256),  
    transforms.CenterCrop(224),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])  
  
# 加载并预处理图像  
img_path = './MEFLUT-resnet/test/bright/video2_frame_00555.jpg'  # 替换为你的图像路径
# img_path = 'E:\\hdr_data\\HDR_CODE\\CLUT-main\\resnet_classifier\\MEFLUT-resnet\\test\\dark\\0.jpg'  # 替换为你的图像路径  
img = Image.open(img_path).convert('RGB')  
img_t = preprocess(img)  
img_t = img_t.unsqueeze(0)  # 增加一个batch维度  
  
# 使用模型进行预测  
with torch.no_grad():  
    outputs = model(img_t)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]  # 计算softmax概率  
    _, predicted = torch.max(outputs, 1)  # 获取预测的类别  
  
# 输出预测结果
print(f'Class probabilities: {probs.numpy()}')
print(f'Predicted class: {predicted.item()}')  