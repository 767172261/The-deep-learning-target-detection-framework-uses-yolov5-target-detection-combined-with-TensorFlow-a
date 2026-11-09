深度学习目标检测框架中利用yolov8进行训练
——太阳能电池片单晶硅数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dcfec2d3620c4546b8fe6de712c788a8.png)
识别——5类太阳能电池片单晶硅数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/31475ed39d0b4209ba68e18bd8567df9.png)

通过工业CCD相机自行采集
拥有图像1108张，图片大小是600x600![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/166e677154554e5ab9ba350f714a6470.png)


5个类别
VOC格式和YOLO格式

目前YOLOv5s识别的map50为0.781

数据集是原始数据集，未做任何清晰化处理![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/27bee863e5aa4a618bc5c70aeff95a62.png)


可直接训练，验证，和测试

构建一个用于太阳能电池片单晶硅缺陷检测的深度学习模型。使用YOLOv5进行目标检测，并利用TensorFlow和PyTorch来实现这一目标。以下是详细的步骤：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/04f396474779474e897ac6c348d04571.png)


1. **环境准备**：
   - 安装必要的库。
   - 下载并组织数据集。

2. **数据预处理**：
   - 将数据集分为训练集、验证集和测试集。
   - 格式化标签文件以便于YOLOv5使用。

3. **模型定义与训练**：
   - 使用YOLOv5进行目标检测。

4. **评估与可视化**：
   - 评估模型性能。
   - 可视化结果。

### 环境准备

首先，我们需要安装必要的库。您可以使用以下命令来设置环境：

```bash
pip install torch torchvision torchaudio pyyaml opencv-python-headless seaborn pandas matplotlib scikit-learn lxml
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 数据预处理

假设您的数据集已经下载并存储在`datasets/solar_cells`目录中，其中包含`images`和`annotations`子目录。`annotations`子目录中的每个`.xml`文件对应一个图像文件，格式为VOC XML格式。

那么就
接下来，我们将编写一个脚本来检查数据集的有效性，并将VOC XML标签转换为YOLOv5所需的格式。

```python
[<title="Data Preparation Script for Solar Cell Defect Detection">]
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random
import numpy as np

# Define paths
base_path = Path('datasets/solar_cells')
images_dir = base_path / 'images'
annotations_dir = base_path / 'annotations'

# Create train, val, test directories
train_images_dir = base_path / 'train/images'
train_labels_dir = base_path / 'train/labels'
val_images_dir = base_path / 'val/images'
val_labels_dir = base_path / 'val/labels'
test_images_dir = base_path / 'test/images'
test_labels_dir = base_path / 'test/labels'

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Function to convert VOC XML to YOLO format
def convert_voc_to_yolo(xml_file, image_width, image_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_lines = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        # Map class names to IDs
        class_id_map = {
            'class1': 0,
            'class2': 1,
            'class3': 2,
            'class4': 3,
            'class5': 4
        }
        class_id = class_id_map[label]
        yolo_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
        yolo_lines.append(yolo_line)
    
    return ''.join(yolo_lines)

# Split dataset into train, val, test
image_files = list(images_dir.glob('*.jpg'))  # Adjust extension if necessary
random.shuffle(image_files)

split_index_1 = int(0.8 * len(image_files))
split_index_2 = int(0.9 * len(image_files))

train_files = image_files[:split_index_1]
val_files = image_files[split_index_1:split_index_2]
test_files = image_files[split_index_2:]

def copy_and_convert(files, src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir):
    for img_file in files:
        ann_file = src_ann_dir / (img_file.stem + '.xml')
        if not ann_file.exists():
            print(f"Missing annotation file for {img_file.name}")
            continue
        
        # Copy image
        shutil.copy(img_file, dst_img_dir)
        
        # Convert and save label
        tree = ET.parse(ann_file)
        root = tree.getroot()
        size = root.find('size')
        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)
        
        yolo_content = convert_voc_to_yolo(ann_file, image_width, image_height)
        with open(dst_ann_dir / (img_file.stem + '.txt'), 'w') as f:
            f.write(yolo_content)

copy_and_convert(train_files, images_dir, annotations_dir, train_images_dir, train_labels_dir)
copy_and_convert(val_files, images_dir, annotations_dir, val_images_dir, val_labels_dir)
copy_and_convert(test_files, images_dir, annotations_dir, test_images_dir, test_labels_dir)

# Create YOLOv5 configuration file
config = {
    "train": str(train_images_dir),
    "val": str(val_images_dir),
    "nc": 5,
    "names": ['class1', 'class2', 'class3', 'class4', 'class5']
}

with open(base_path / 'solar_cells.yaml', 'w') as f:
    import yaml
    yaml.dump(config, f)

print("Dataset prepared and YOLOv5 config created.")
```

### 模型定义与训练

我们将使用YOLOv5进行目标检测。以下是训练脚本 `train_detection.py`:

```python
[<title="Training Script for Solar Cell Defect Detection using YOLOv5">]
import subprocess

# Run YOLOv5 training command
command = [
    'python', 'train.py',
    '--img', '640',          # Image size
    '--batch', '16',         # Batch size
    '--epochs', '50',        # Number of epochs
    '--data', '../datasets/solar_cells/solar_cells.yaml',  # Path to dataset YAML file
    '--cfg', './models/yolov5s.yaml',            # Model configuration
    '--weights', 'yolov5s.pt',                     # Pretrained weights
    '--project', '../runs/train',                 # Project directory
    '--name', 'solar_cell_defect_detection'       # Experiment name
]

subprocess.run(command)
```

### 评估与可视化

使用YOLOv5自带的评估脚本来评估目标检测模型。

```python
[<title="Evaluation Script for Solar Cell Defect Detection using YOLOv5">]
import subprocess

# Run YOLOv5 evaluation command
command = [
    'python', 'val.py',
    '--data', '../datasets/solar_cells/solar_cells.yaml',  # Path to dataset YAML file
    '--weights', '../runs/train/solar_cell_defect_detection/weights/best.pt',  # Path to best model weights
    '--img', '640',                             # Image size
    '--conf', '0.5',                            # Confidence threshold
    '--iou-thres', '0.45',                      # IoU threshold
    '--task', 'val',                            # Task type
    '--save-json',                               # Save JSON results
    '--project', '../runs/val',                   # Project directory
    '--name', 'solar_cell_defect_detection_val'  # Experiment name
]

subprocess.run(command)
```

### 用户界面

我们将使用 PyQt5 创建一个简单的 GUI 来加载和运行模型进行实时预测。以下是用户界面脚本 `ui.py`:

```python
[<title="PyQt5 Main Window for Solar Cell Defect Detection">]
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import ultralytics.yolo.engine.model

# Load model
detection_model = ultralytics.yolo.engine.model.Model(cfg='./models/yolov5s.yaml').load('../runs/train/solar_cell_defect_detection/weights/best.pt')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Solar Cell Defect Detection System")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)

        self.start_prediction_button = QPushButton("Start Prediction", self)
        self.start_prediction_button.clicked.connect(self.start_prediction)
        self.layout.addWidget(self.start_prediction_button)

        self.stop_prediction_button = QPushButton("Stop Prediction", self)
        self.stop_prediction_button.clicked.connect(self.stop_prediction)
        self.layout.addWidget(self.stop_prediction_button)

        self.central_widget.setLayout(self.layout)

        self.image_path = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.display_image(file_name)

    def display_image(self, path):
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

    def start_prediction(self):
        if self.image_path is not None and not self.timer.isActive():
            self.timer.start(30)  # Update frame every 30 ms

    def stop_prediction(self):
        if self.timer.isActive():
            self.timer.stop()
            self.image_label.clear()

    def update_frame(self):
        original_image = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Detection
        detection_results = detection_model(image_rgb, size=640)[0].boxes.data.cpu().numpy()
        for result in detection_results:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)

            # Map class ID to name
            class_names = ['class1', 'class2', 'class3', 'class4', 'class5']
            class_name = class_names[cls]

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_rgb, f'{class_name} ({conf:.2f})', (x1, y1 - 10), font, 0.9, (0, 255, 0), 2)

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

请确保将路径替换为您实际的路径。

### 使用说明

1. **配置路径**：
   - 确保 `datasets/solar_cells` 目录结构正确，并且包含 `images` 和 `annotations` 子目录。
   - 确保 `runs/train/solar_cell_defect_detection/weights/best.pt` 是训练好的 YOLOv5 模型权重路径。

2. **运行脚本**：
   - 在终端中运行 `data_preparation.py` 脚本来检查数据集的有效性并创建 YOLOv5 配置文件。
   - 在终端中运行 `train_detection.py` 脚本来训练目标检测模型。
   - 在终端中运行 `evaluate_detection.py` 来评估目标检测模型性能。
   - 在终端中运行 `ui.py` 来启动 GUI 应用程序。

3. **注意事项**：
   - 确保所有必要的工具箱已安装，特别是 TensorFlow 和 PyQt5。
   - 根据需要调整参数，如 `epochs` 和 `batch_size`。

### 示例

假设您的数据文件夹结构如下：
```
datasets/
└── solar_cells/
    ├── images/
    └── annotations/
```

并且每个数据集中包含相应的图像和标签文件。运行 `ui.py` 后，您可以点击按钮来加载图像并进行太阳能电池片单晶硅缺陷检测。


