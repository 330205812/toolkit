"""
1.写一个数据处理类
    需要的功能有：通过传入切块后的图像大小，切块的偏移量来对图像进行切块、切块超出边缘时回退
               保存数据标签

    需要传入的参数：图像路径、标签路径、切块大小、偏移量大小、保存路径（图像和标签的路径分别用jion的方式生成在同一级目录下）
                 标签数据保存的格式（可扩展功能）、图像保存的格式
"""
import json
import os.path
from collections import defaultdict

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class SAMdatapipeline(Dataset):
    """
    img_dir：图像路径
    annotation_dir：标签路径
    block_size：切片窗口大小
    offset：窗口滑动偏移量
    out_dir：输出路径
    category：可以从数据集中挑选指定类别重新生成数据集
    """
    def __init__(self,
                 img_dir,
                 annotation_dir,
                 block_size,
                 offset,
                 out_dir,
                 category="plane",
                 **kwargs):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.block_size = block_size
        self.offset = offset
        self.out_dir = out_dir
        self.category = category
        self.category_id = None
        self.mapping_imgid_dict = defaultdict(list)
        self.mapping_ann_dict = defaultdict(list)

    def __getitem__(self, index):
        """
        返回三个值，图像，图像中的bbox标签，图像中的mask标签
        """
        json_path = os.path.join(r'D:\data\iSAID\plane', 'plane_train.json')  # 你的标签路径
        with open(json_path, 'r') as ann_file:
            ann_data = json.load(ann_file)
        img_list = [img["file_name"] for img in ann_data["images"]]
        img_id_list = [img["id"] for img in ann_data["images"]]
        img_file_name = img_list[index]
        img_id = img_list[index]
        img_file = os.path.join(self.img_dir, img_file_name)
        img = cv2.imread(img_file)
        bbox = [ann["bbox"] for ann in ann_data["annotations"] if ann["image_id"] == img_id]
        mask = [ann["segmentation"] for ann in ann_data["annotations"] if ann["image_id"] == img_id]
        return img, bbox, mask

    def load_data_(self):
        """
        从coco格式的json数据集中提取指定类别的标签，并生成json文件
        """
        json_path = os.path.join(self.annotation_dir, 'iSAID_train_20190823_114751.json')  # 标签路径
        with open(json_path, 'r') as ann_file:
            ann_data = json.load(ann_file)
        if self.category:
            target = self.category
            category_id = next((d["id"] for d in ann_data["categories"] if d["name"] == target), None)
            self.category_id = category_id
            # 用image_id作为键索引保存对应的
            mapping_ann_dict = defaultdict(list)
            for d_ in ann_data["annotations"]:
                if d_["category_id"] == category_id:
                    mapping_ann_dict[d_["image_id"]].append(d_)

            mapping_imgid_dict = defaultdict(list)
            for d_ in mapping_ann_dict:
                target_dict = next(info for info in ann_data["images"] if info["id"] == d_)
                num_boxes = len(mapping_ann_dict[d_])
                target_dict["num_boxes"] = num_boxes
                mapping_imgid_dict[d_].append(target_dict)
        else:
            # 用image_id作为键索引保存对应的
            mapping_ann_dict = defaultdict(list)
            for d_ in ann_data["annotations"]:
                mapping_ann_dict[d_["image_id"]].append(d_)

            mapping_imgid_dict = defaultdict(list)
            for d_ in ann_data["images"]:
                num_boxes = len(mapping_ann_dict[d_["id"]])
                d_["num_boxes"] = num_boxes
                mapping_imgid_dict[d_["id"]].append(d_)

        self.mapping_imgid_dict = mapping_imgid_dict
        self.mapping_ann_dict = mapping_ann_dict

    def block(self):
        """
        将图像切块成指定大小并生成新的标签
        """
        image_info_dict = self.mapping_imgid_dict
        image_ann_dict = self.mapping_ann_dict
        # 保存转化后的标签
        targetobject_dict = defaultdict(list)
        count = 0
        # 从image_info_dict循环读取图片，一张一张图片进行处理
        image_id = 0
        for i in tqdm.tqdm(image_info_dict):
            file_name = image_info_dict[i][0]["file_name"]
            img_file = os.path.join(self.img_dir, file_name)
            # 加载图像
            image = cv2.imread(img_file)
            # 获取图像的高度和宽度
            height, width = image.shape[:2]
            # 计算分割的行数和列数
            rows = height // self.offset
            cols = width // self.offset

            # 对一张图像进行切块操作
            for i_ in range(rows):
                for j_ in range(cols):
                    # 计算分割块的起始和结束坐标
                    start_y = i_ * self.offset
                    end_y = start_y + self.block_size
                    start_x = j_ * self.offset
                    end_x = start_x + self.block_size
                    if end_x >= width:
                        start_x = start_x - (end_x - width)
                        end_x = width
                        if start_x < 0:
                            start_x = 0
                            # end_x = width
                    if end_y >= height:
                        start_y = start_y - (end_y - height)
                        end_y = height
                        if start_y < 0:
                            start_y = 0
                            # end_y = height

                    # 第一步：提取分割图像
                    image_patch = image[start_y:end_y, start_x:end_x]
                    block_bbox_list = []
                    temp_info_list = []
                    for b in range(len(image_ann_dict[i])):
                        box = image_ann_dict[i][b]["bbox"]
                        xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
                        bbox = [xmin, ymin, xmax, ymax]
                        start = [start_x, start_y, start_x, start_y]
                        # 判断bbox是否在切片内
                        if xmin >= start_x and ymin >= start_y and xmax <= end_x and ymax <= end_y:
                            # 将对应原图的bbox坐标转化为对应切块后的坐标
                            block_bbox = [b - s for b, s in zip(bbox, start)]
                            block_bbox_list.append(block_bbox)
                            # 对mask坐标进行修改
                            segmentation = image_ann_dict[i][b]["segmentation"]
                            dx = start_x
                            dy = start_y
                            new_segmentation = [
                                [coord - dx if idx % 2 == 0 else coord - dy
                                 for idx, coord in enumerate(inner_list)] for
                                inner_list in segmentation]

                            temp_info_dict = {'segmentation': new_segmentation,
                                              'category_id': image_ann_dict[i][b]['category_id'],
                                              'category_name': image_ann_dict[i][b]['category_name'],
                                              "iscrowd": image_ann_dict[i][b]["iscrowd"],
                                              "area": image_ann_dict[i][b]["area"],
                                              "bbox": block_bbox}
                            temp_info_list.append(temp_info_dict)

                    # 第二步： 判断切块图像中是否含有目标,有的话保存对应的images和annotations信息
                    if len(block_bbox_list) != 0:
                        # 有目标，先保存切块图像
                        out_path = os.path.join(self.out_dir, self.category)
                        block_file_name = os.path.join(out_path, file_name.split('.')[0])
                        cv2.imwrite(f'{block_file_name}_{i_}_{j_}.jpg', image_patch)
                        # 保存对应的images和annotation信息
                        info_dict = {"id": image_id,
                                     "file_name": os.path.basename(f'{block_file_name}_{i_}_{j_}.jpg')}

                        targetobject_dict["images"].append(info_dict)
                        category_dict = {"id": self.category_id,
                                         "name": self.category}
                        if category_dict not in targetobject_dict["categories"]:
                            targetobject_dict["categories"].append(category_dict)
                        for info_dict in temp_info_list:
                            ann_dict = {"id": count,
                                        "image_id": image_id,
                                        "segmentation": info_dict['segmentation'],
                                        "category_id": info_dict["category_id"],
                                        "category_name": info_dict["category_name"],
                                        "iscrowd": info_dict["iscrowd"],
                                        "area": info_dict["area"],
                                        "bbox": info_dict["bbox"]}
                            count += 1
                            # 对mask坐标进行修改
                            targetobject_dict["annotations"].append(ann_dict)
                        image_id += 1

        # 第三步：将标签信息写入文件
        targetobject_dict = dict(targetobject_dict)
        with open(r"D:\data\iSAID\plane\plane_train.json", "w") as f:
            json.dump(targetobject_dict, f)

    def show_box(self):
        """
        保存数据集可视化结果
        """
        pass

    def show_mask(self):
        """
        保存数据集可视化结果
        """
        pass


def save():
    """
    coco格式的数据集可视化
    """
    # 初始化COCO对象
    coco = COCO(r'D:\data\iSAID\plane\plane_train.json')

    # 获取图像ID
    img_ids = coco.getImgIds()

    for i in img_ids:
        # 选择一个图像ID
        img_id = img_ids[i]

        # 加载图像信息
        img_info = coco.loadImgs(img_id)[0]

        # 读取图像
        img = cv2.imread('D:/data/iSAID/plane/' + img_info['file_name'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取该图像的所有注释ID
        ann_ids = coco.getAnnIds(imgIds=img_info['id'])

        # 加载注释
        anns = coco.loadAnns(ann_ids)

        # 创建一个新的matplotlib图像
        fig, ax = plt.subplots(1)

        # 显示图像
        ax.imshow(img)

        # 遍历每个注释
        for ann in anns:
            # 如果注释有bbox
            if 'bbox' in ann:
                bbox = ann['bbox']
                # 创建一个矩形patch 要传入左上角的点和w,h
                x = bbox[0]
                y = bbox[1]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                # 添加patch到图像上
                ax.add_patch(rect)

            # 如果注释有segmentation
            if 'segmentation' in ann:
                # 获取segmentation
                seg = ann['segmentation']
                # 如果segmentation是一个列表
                if isinstance(seg, list):
                    # 遍历每个segmentation
                    for polygon in seg:
                        # 创建一个polygon patch
                        poly = patches.Polygon(np.array(polygon).reshape((int(len(polygon) / 2), 2)), fill=False)
                        # 添加patch到图像上
                        ax.add_patch(poly)

        # 保存图像到本地
        output_path = 'D:/data/iSAID/plane_show/' + img_info['file_name']
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

        # 关闭图像
        plt.close()


if __name__ == '__main__':
    img_dir_ = r'D:\data\DOTA\data\train\images'
    annotation_dir_ = r'D:\data\iSAID\train\Annotations'
    block_size_ = 1024
    offset_ = 512
    out_dir_ = r'D:\data\iSAID'
    data = SAMdatapipeline(img_dir_, annotation_dir_, block_size_, offset_, out_dir_)
    image, bboxes, masks = data[1]
    print("-------------")
    # data.load_data_()
    # data.block()
    # save()
