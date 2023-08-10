import copy
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def convert_to_rotated_box(xmin, ymin, xmax, ymax):
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    theta = 0
    return cx, cy, w, h, theta


def convert_to_corners(cx, cy, h, w, angle):
    """
    (cx, cy):是旋转框的中心点
    h, w：h是长边长度，w是短边长度
    angle：角度
    """
    hx, hy = h / 2 * np.cos(angle), h / 2 * np.sin(angle)
    wx, wy = -w / 2 * np.sin(angle), w / 2 * np.cos(angle)
    p1 = (cx - hx - wx, cy - hy - wy)
    p2 = (cx + hx - wx, cy + hy - wy)
    p3 = (cx + hx + wx, cy + hy + wy)
    p4 = (cx - hx + wx, cy - hy + wy)
    points = [p1, p2, p3, p4]
    return points


def convert_xml_to_txt(xml_path, txt_path):
    """
    将xml格式的旋转框数据集标签转换为txt格式的标签
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    with open(txt_path, 'w') as txt_file:
        for object in root.findall('object'):
            robndbox = object.find('robndbox')
            cx = robndbox.find('cx').text
            cy = robndbox.find('cy').text
            h = robndbox.find('h').text
            w = robndbox.find('w').text
            angle = robndbox.find('angle').text
            points = convert_to_corners(float(cx), float(cy), float(h), float(w), float(angle))
            x1, y1 = points[0][0], points[0][1]
            x2, y2 = points[1][0], points[1][1]
            x3, y3 = points[2][0], points[2][1]
            x4, y4 = points[3][0], points[3][1]

            class_id = object.find('name').text
            line = f'{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {class_id} 0\n'
            txt_file.write(line)


def convert_txt_to_txt(targeet_path, out_path):
    """
    将txt格式的水平框数据集标签转换为txt格式的旋转框格式标签
    """
    # 使用字典存储每张图像的所有目标框
    boxes_dict = defaultdict(list)
    for filename in os.listdir(targeet_path):
        if filename.endswith('.txt'):
            # 读取文件
            with open(os.path.join(targeet_path, filename), 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip().split()
                # cls = line[0]
                cls = line[4]
                # xmin, ymin, xmax, ymax = map(int, map(float, line[1:5]))
                xmin, ymin, w, h = map(int, map(float, line[:4]))
                xmax = xmin + w
                ymax = ymin + h
                boxes_dict[filename].append((xmin, ymin, xmax, ymax, cls))
                cx, cy, h, w, angle = convert_to_rotated_box(xmin, ymin, xmax, ymax)
                points = convert_to_corners(float(cx), float(cy), float(h), float(w), float(angle))
                x1, y1 = points[0][0], points[0][1]
                x2, y2 = points[1][0], points[1][1]
                x3, y3 = points[2][0], points[2][1]
                x4, y4 = points[3][0], points[3][1]

                # class_id = object.find('name').text
                line = f'{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {cls} 0\n'

                with open(os.path.join(out_path, filename), 'a') as txt_file:
                    txt_file.write(line)


def xml2txt():
    """
    标签文件格式转换方法
    """
    xml_dir = r'E:\RSDD-SAR\Annotations'  # 替换为你的XML文件夹路径
    txt_dir = r'E:\RSDD-SAR\Annotations_txt'  # 替换为你希望保存TXT文件的文件夹路径

    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            txt_path = os.path.join(txt_dir, xml_file.replace('.xml', '.txt'))
            convert_xml_to_txt(xml_path, txt_path)


def draw_rotated_box(img, points, class_name, color=(0, 255, 0)):
    """
    通过四个顶点来画出旋转框
    """
    # 确保您的四个点的坐标是整数
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    # 添加类别信息
    cv2.putText(img, str(class_name), (points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def visualize_rotated(image_dir, label_dir, output_dir):
    """
    可视化旋转框
    标签文件格式为xml
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(label_dir):
        if filename.endswith('.xml'):
            # 读取图像
            img_path = os.path.join(image_dir, filename.replace('.xml', '.jpg'))
            img = cv2.imread(img_path)

            # 读取标注文件
            tree = ET.parse(os.path.join(label_dir, filename))
            root = tree.getroot()
            for object in root.findall('object'):
                class_name = object.find('name').text
                bndbox = object.find('robndbox')
                cx = float(bndbox.find('cx').text)
                cy = float(bndbox.find('cy').text)
                w = float(bndbox.find('w').text)
                h = float(bndbox.find('h').text)
                angle = float(bndbox.find('angle').text)
                points = convert_to_corners(float(cx), float(cy), float(h), float(w), float(angle))

                # 绘制旋转的边界框
                draw_rotated_box(img, points, class_name)

            # 保存图像
            output_path = os.path.join(output_dir, filename.replace('.xml', '.jpg'))
            cv2.imwrite(output_path, img)


def count_classes_in_files(directory):

    # 初始化一个默认字典来存储类别和数量
    class_counts = defaultdict(int)
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # 读取文件
            with open(os.path.join(directory, filename), 'r') as f:
                lines = f.readlines()

            # 遍历文件的每一行
            for line in lines:
                # 分割行并获取类别
                class_id = line.strip().split(' ')[-2]
                # 更新类别的数量
                class_counts[class_id] += 1

    return class_counts


def visualize_boxes(txt_path, img_dir, output_dir):
    # 使用字典存储每张图像的所有目标框
    boxes_dict = defaultdict(list)
    for filename in os.listdir(txt_path):
        if filename.endswith('.txt'):
            # 读取文件
            with open(os.path.join(txt_path, filename), 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip().split()
                cls = line[0]
                xmin, ymin, xmax, ymax = map(int, line[1:5])
                boxes_dict[filename].append((xmin, ymin, xmax, ymax, cls))

            for filename, boxes in boxes_dict.items():
                img = cv2.imread(os.path.join(img_dir, filename.replace('.txt', '.jpg')))

                for box in boxes:
                    xmin, ymin, xmax, ymax, cls = box
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(img, cls, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                cv2.imwrite(os.path.join(output_dir, filename.replace('.txt', '.jpg')), img)


def aug_data(outpath, img_dir):
    image_dict = {'image_name': '000009.jpg 000157.jpg 000177.jpg 000180.jpg 000187.jpg 000273.jpg 000333.jpg '
                                '000345.jpg 000540.jpg 000568.jpg 000570.jpg 000674.jpg 000682.jpg 000715.jpg '
                                '000724.jpg 000752.jpg 000804.jpg 000805.jpg 000821.jpg 000912.jpg 000914.jpg '
                                '000915.jpg 000918.jpg '}

    # 使用字典存储每张图像的所有目标框
    boxes_dict = defaultdict(list)
    image_names = image_dict['image_name'].split()
    # 打印结果
    count = 0
    for image_name in image_names:
        # 加载图像
        image = cv2.imread(os.path.join(img_dir, image_name))
        new_image = image.transpose((2, 0, 1))

        label_name = image_name.replace('.jpg', '.txt')
        with open(os.path.join(outpath, label_name), 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            cls = line[0]
            xmin, ymin, xmax, ymax = map(int, map(float, line[1:5]))
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin

            # x, y是边界框的左上角坐标，w, h是边界框的宽度和高度
            boxes_dict[image_name].append([x, y, w, h, cls])

        # 假设你已经获取到了所有目标的边界框
        bboxes = boxes_dict[image_name]  # 这是一个列表，包含了所有目标的边界框
        aug_boxes_dict = copy.deepcopy(bboxes)  # 这个列表，用来保存增强后所有目标的边界框
        for i in boxes_dict[image_name]:
            # 提取目标
            x, y, w, h, cls = i[0], i[1], i[2], i[3], i[4]
            target = image[y:y + h, x:x + w]

            # 确定粘贴的位置
            while True:
                paste_x, paste_y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])  # 随机选择一个位置
                if not any((paste_x < box[0] + box[2] and paste_x + w > box[0] and paste_y < box[1] + box[3] and
                            paste_y + h > box[1]) for box in aug_boxes_dict):
                    break  # 如果这个位置没有其他目标，那么就跳出循环

            # 如果需要，调整目标的大小
            if paste_x + w > image.shape[1] or paste_y + h > image.shape[0]:
                target = cv2.resize(target, (min(w, image.shape[1] - paste_x), min(h, image.shape[0] - paste_y)))

            # 粘贴目标
            image[paste_y:paste_y + target.shape[0], paste_x:paste_x + target.shape[1]] = target
            aug_boxes_dict.append([paste_x, paste_y, target.shape[1], target.shape[0], cls])

        # 保存图像
        target_file = r'D:\data\match\ship_preliminary_contest\ship_train\aug_images'
        target_image = os.path.join(target_file, str(count) + image_name)
        count += 1
        cv2.imwrite(target_image, image)

        # 保存标签
        target_label_file = r'D:\data\match\ship_preliminary_contest\ship_train\aug_labels'
        target_label_name = os.path.join(target_label_file, label_name)
        with open(target_label_name, 'a') as txt_file:
            for line in aug_boxes_dict:
                line_ = ' '.join(map(str, line)) + '\n'
                txt_file.write(line_)


def denoising(image_path, out_path):
    """
    图像直方图可视化
    """
    for i, filename in enumerate(os.listdir(image_path)):
        print(filename)
        image_ = os.path.join(image_path, filename)
        # 加载图像
        image = cv2.imread(image_, cv2.IMREAD_GRAYSCALE)  # 假设你的SAR图像是灰度图

        # 计算直方图
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        # 创建新的图形
        plt.figure(i)
        # 绘制直方图
        plt.plot(hist)
        # 保存直方图
        filename_ = 'hist_' + filename
        hist_name = os.path.join(out_path, filename_)
        plt.savefig(hist_name)


if __name__ == '__main__':
    # root = 'E:/RSDD-SAR'
    # xml2txt()
    image_dir = r'E:\RSDD-SAR\JPEGImages'  # 替换为你的图像文件夹路径
    label_dir = r'E:\RSDD-SAR\Annotations'  # 替换为你的标注文件夹路径
    output_dir = r'E:\RSDD-SAR\visual_data'  # 替换为你希望保存可视化图像的文件夹路径
    visualize_rotated(image_dir, label_dir, output_dir)
    # class_counts = count_classes_in_files(os.path.join(root, 'Annotations_txt'))
    # print(class_counts)
    txt_path = r'D:\data\match\ship_preliminary_contest\ship_train\labels'
    img_dir = r'D:\data\match\ship_preliminary_contest\ship_train\images'
    output_dir = r'D:\data\match\ship_preliminary_contest\visualize_boxes'
    # 可视化水平框
    # visualize_boxes(txt_path, img_dir, output_dir)
    # 1.转换标签
    target_path = r'D:\data\match\ship_preliminary_contest\ship_train\aug_labels'
    outpath = r'D:\data\match\ship_preliminary_contest\ship_train\cnvert_aug_labels'
    # convert_txt_to_txt(target_path, outpath)
    # 数据增强
    # aug_data(txt_path, img_dir)
    # 绘制图像的直方图
    # hist_path = r'D:\data\match\ship_preliminary_contest\ship_train\hist'
    # denoising(img_dir, hist_path)
