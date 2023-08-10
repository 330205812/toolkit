import os
import cv2
from tqdm import tqdm

# 定义文件夹路径和文件后缀
images_folder = r"I:\match\match3\train_data\images"
labels_folder = r"I:\match\match3\train_data\labelTxt_new"
output_folder = r"I:\match\match3\train_data\images_draw"
file_extension = ".txt"

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历txt文件
for filename in tqdm(os.listdir(labels_folder)):
    if filename.endswith(file_extension):
        # 获取图像文件名和路径
        image_filename = os.path.splitext(filename)[0]
        # 需要改的地方
        image_path = os.path.join(images_folder, image_filename + '.bmp')

        # 读取图像
        image = cv2.imread(image_path)

        # 读取txt文件
        label_file = os.path.join(labels_folder, filename)
        with open(label_file, "r") as f:
            lines = f.readlines()

        # 遍历每行内容
        if len(lines) != 0:
            for line in lines:
                # 解析行数据
                data = line.strip().split()
                x1, y1, x2, y2, x3, y3, x4, y4, class_name, _ = data

                # 转换为浮点数
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, [x1, y1, x2, y2, x3, y3, x4, y4])

                # 绘制边界框
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), (0, 255, 0), 2)
                cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
                cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), (0, 255, 0), 2)

                # 在边界框上方绘制类别名称
                text = class_name
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_origin = (int(x1), int(y1) - 10)
                cv2.rectangle(image, (int(x1), int(y1) - text_size[1] - 10),
                              (int(x1) + text_size[0], int(y1) - 10), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, text, text_origin, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # 保存绘制完的图像到输出文件夹
        output_path = os.path.join(output_folder, image_filename + '.jpg')
        cv2.imwrite(output_path, image)
