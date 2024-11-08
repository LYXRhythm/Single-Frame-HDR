# HDR处理系统
  实现目标：将图片经过HDR处理后色彩，饱和度等更加鲜艳，生动。
  
  ![2024-11-08 15-06-07 的屏幕截图](https://github.com/user-attachments/assets/91922739-f80d-4d69-a85a-8c7b3fb3949f)
## 使用示范
需在Linux配置环境按要求启动 
### 安装环境
```
pip install -r requirements.txt
```
### 数据集获取

### 启动程序
选择数据集，数据集下有input_test,target_test。选择权重文件夹，要求权重文件夹下存在dark.pth,bright.pth。示例选择all_pth。
```
python qt_test.py
```

