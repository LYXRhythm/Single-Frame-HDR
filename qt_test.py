import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, \
    QMainWindow, QScrollArea, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from torchvision.utils import save_image
import numpy as np
from test import test  # 假设原代码在 `test.py` 中
from models import *
from datasets import *
from utils.losses import *
from torch.utils.data import DataLoader
from os.path import join
import os
from parameters import *
import time
from torchvision import models
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class TestWorker(QThread):
    update_input_img = pyqtSignal(QPixmap)
    update_output_img = pyqtSignal(QPixmap)
    update_psnr = pyqtSignal(str)  # 用于更新PSNR结果显示

    def __init__(self,model1,model2,test_dataloader, device, data_root, current_batch_idx=0):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.test_dataloader = test_dataloader
        self.device = device
        self.data_root = data_root  # 保存当前数据集路径
        self._is_running = True  # 控制线程运行的标志
        self.current_batch_idx = current_batch_idx  # 记录当前处理的批次
        self.last_data_root = None  # 用于检测数据集是否改变
    def run(self):
        avg_psnr_out = 0
        avg_psnr_normalized = 0
        self.save_path = join(self.data_root,'output')
        os.makedirs(self.save_path, exist_ok=True)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        k = 0
        for i, batch in enumerate(self.test_dataloader):
            if not self._is_running:
                break  # 如果线程已停止，则退出循环
            # 跳过已经处理过的批次
            if i < self.current_batch_idx:
                continue

            inputs = batch["input"].to(self.device)
            targets_org = batch["target_org"].to(self.device)
            inputs_org = batch.get("input_org").to(self.device)
            name = os.path.splitext(batch["name"][0])[0]
            class_input = F.interpolate(inputs_org, size=(224, 224), mode='bilinear', align_corners=False)
            model_path = 'classifier.pth'
            picture_path = batch["input_path"]
            img_path = picture_path  # 替换为你的图像路径
            # img_path = 'E:\\hdr_data\\HDR_CODE\\CLUT-main\\resnet_classifier\\MEFLUT-resnet\\test\\dark\\0.jpg'  # 替换为你的图像路径
            img = Image.open(img_path[0]).convert('RGB')
            img_t = preprocess(img)
            img_t = img_t.unsqueeze(0)  # 增加一个batch维度

            model3 = models.resnet18(pretrained=False)  # 不使用预训练的权重
            num_ftrs = model3.fc.in_features  # 获取全连接层的输入特征数
            model3.fc = torch.nn.Linear(num_ftrs, 2)  # 修改全连接层为二分类
            model3.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))  # 加载模型权重
            model3.eval()  # 设置为评估模式
            # 使用模型进行预测
            with torch.no_grad():
                outputs = model3(img_t)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]  # 计算softmax概率
                _, predicted = torch.max(outputs, 1)  # 获取预测的类别
                #print(predicted)
            if predicted == 0:
                self.model = self.model2
            else:
                self.model = self.model1

            self.model.eval()
            results = self.model(inputs, inputs_org, TVMN=None)
            fakes = results["fakes"]
            psnr_out = psnr(fakes, targets_org).item()
            psnr_normalized = calculate_normalized_psnr(fakes, targets_org, torch.max(targets_org))

            avg_psnr_out += psnr_out
            avg_psnr_normalized += psnr_normalized

            input_img = self.tensor_to_qpixmap(inputs_org, target_size=(1920, 1080))
            output_img = self.tensor_to_qpixmap(fakes, target_size=(1920, 1080))

            self.update_input_img.emit(input_img)
            self.update_output_img.emit(output_img)

            self.current_batch_idx = i + 1  # 更新当前批次索引

            fakes_img = fakes.squeeze().data

            if fakes_img.shape[0] > 3:
                fakes_img = fakes_img.permute(2, 0, 1)
            save_file_path = join(self.save_path, f"{name}.jpg")
            #print(save_file_path)
            save_image(fakes_img, save_file_path)
            print("PSNR-L {:>0.2f}dB;PSNR-u results: {:>0.2f}dB;".format(psnr_out, psnr_normalized))
            time.sleep(2)

        avg_psnr_out /= len(self.test_dataloader)
        avg_psnr_normalized /= len(self.test_dataloader)
        #avg_psnr_out /= k
        #avg_psnr_normalized /= k
        psnr_results = "Test PSNR-L results: {:>0.2f}dB; Test PSNR-u results: {:>0.2f}dB;".format(avg_psnr_out, avg_psnr_normalized)
        new_folder_name = self.save_path + f" {avg_psnr_out:.2f}dB"
        os.rename(self.save_path, new_folder_name)
        self.update_psnr.emit(psnr_results)
        self.finished.emit()

    def tensor_to_qpixmap(self, tensor, target_size=None):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor = tensor.squeeze(0)

        array = tensor.detach().numpy()
        array = np.transpose(array, (1, 2, 0))

        array = (array * 255).clip(0, 255).astype(np.uint8)

        if target_size:
            from PIL import Image
            image = Image.fromarray(array)
            image = image.resize(target_size)
            array = np.array(image)

        h, w, ch = array.shape
        bytes_per_line = ch * w
        image = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(image)
        return pixmap

    def stop(self):
        self._is_running = False  # 设置为False，停止线程


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDR处理系统")
        self.setGeometry(100, 100, 1000, 600)  # 增大窗口尺寸

        # UI Components
        self.select_data_button = QPushButton("选择数据集路径")
        self.select_data_button.setFixedSize(200, 40)  # 设置按钮大小
        self.select_data_button.clicked.connect(self.select_data_path)

        self.select_model_button = QPushButton("选择HDR权重")
        self.select_model_button.setFixedSize(200, 40)  # 设置按钮大小
        self.select_model_button.clicked.connect(self.select_model_path)

        self.start_button = QPushButton("开始HDR处理")
        self.start_button.setFixedSize(200, 40)  # 设置按钮大小
        self.start_button.clicked.connect(self.run_test)

        self.psnr_label = QLabel("PSNR Results: ")
        self.input_label = QLabel("原始图")
        self.output_label = QLabel("HDR处理图")
        self.input_label.setScaledContents(True)
        self.output_label.setScaledContents(True)

        # Scroll areas for images
        self.input_scroll = QScrollArea()
        self.input_scroll.setWidget(self.input_label)
        self.input_scroll.setWidgetResizable(True)

        self.output_scroll = QScrollArea()
        self.output_scroll.setWidget(self.output_label)
        self.output_scroll.setWidgetResizable(True)

        # Layout for buttons (Horizontal layout for buttons)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_data_button)
        button_layout.addWidget(self.select_model_button)
        button_layout.addWidget(self.start_button)

        # Wrap the button layout in a QVBoxLayout to fit the remaining layout
        button_layout_wrapper = QVBoxLayout()
        button_layout_wrapper.addLayout(button_layout)

        # Layout for images and PSNR results
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.input_scroll)
        images_layout.addWidget(self.output_scroll)

        psnr_layout = QVBoxLayout()
        psnr_layout.addWidget(self.psnr_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout_wrapper)  # Add the button layout
        main_layout.addLayout(images_layout)
        main_layout.addLayout(psnr_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Model and data paths
        self.model = None
        self.test_dataloader = None
        self.data_root = "E:/BaiduNetdiskDownload/CLUT/CLUT/dataset/MEFLUT-bright"  # 默认数据路径
        self.model_path = None
        self.worker = None
        self.last_data_root = None  # 保存上次的data_root
        self.current_batch_idx = 0  # 当前批次索引

    def select_data_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", self.data_root)
        if folder:
            self.data_root = folder

    def select_model_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Weights Folder")
        if folder:
            dark_path = os.path.join(folder, "dark.pth")
            bright_path = os.path.join(folder, "bright.pth")

            if os.path.exists(dark_path) and os.path.exists(bright_path):
                self.model_path1 = dark_path
                self.model_path2 = bright_path
                print(dark_path)
                print(bright_path)
            else:
                QMessageBox.warning(self, "File Missing", "The folder must contain both dark.pth and bright.pth.")

    def run_test(self):
        hparams = parser.parse_args()
        model_name = "CLUTNet 20+05+20"
        epoch = 362
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 如果没有选择新的权重路径，使用上次的选择
        if not hasattr(self, 'model_path1') or not hasattr(self, 'model_path2'):
            QMessageBox.warning(self, "Warning", "请先选择权重路径！")
            return

        # 加载模型
        model1 = eval('CLUTNet')('20+05+20').to(device)
        model2 = eval('CLUTNet')('20+05+20').to(device)
        ckpt1 = torch.load(self.model_path1,map_location=torch.device('cpu'))
        ckpt2 = torch.load(self.model_path2,map_location=torch.device('cpu'))
        model1.load_state_dict(ckpt1, strict=True)
        model2.load_state_dict(ckpt2, strict=True)

        # 检查是否需要重新加载数据集
        if self.data_root != self.last_data_root or self.test_dataloader is None:
            self.test_dataloader = DataLoader(
                eval('FiveK')(self.data_root, split="test", model='CLUTNet'),
                batch_size=1,
                shuffle=False,
                num_workers=hparams.num_workers,
            )
            self.last_data_root = self.data_root
            self.current_batch_idx = 0  # 重置当前批次索引

        # 创建或重新启动 TestWorker
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()

        self.worker = TestWorker(model1, model2, self.test_dataloader, device, self.data_root, self.current_batch_idx)
        self.worker.update_input_img.connect(self.update_input_image)
        self.worker.update_output_img.connect(self.update_output_image)
        self.worker.update_psnr.connect(self.update_psnr)
        self.worker.finished.connect(self.on_worker_finished)  # 连接线程结束的信号
        self.worker.start()

        self.start_button.setEnabled(False)

    def on_worker_finished(self):
        self.start_button.setEnabled(True)  # 重新启用按钮，允许重新选择数据和权重

    def update_input_image(self, pixmap):
        self.input_label.setPixmap(pixmap)

    def update_output_image(self, pixmap):
        self.output_label.setPixmap(pixmap)

    def update_psnr(self, psnr_result):
        self.psnr_label.setText(psnr_result)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())