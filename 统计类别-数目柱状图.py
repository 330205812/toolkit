import os
import matplotlib.pyplot as plt


def count_objects_in_txt_files(directory):
    class_counts = {}

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            class_counts = update_class_counts(file_path, class_counts)

    return class_counts


def update_class_counts(file_path, class_counts):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            if len(line) >= 9:
                class_name = line[8]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return class_counts


def plot_bar_chart(class_counts):
    # 类别数目最大的在左边
    # sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))
    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: int(item[0]), reverse=False))
    classes = list(sorted_class_counts.keys())
    counts = list(sorted_class_counts.values())
    plt.figure(figsize=(12,10))
    plt.bar(classes, counts)
    plt.xlabel('Class')
    plt.ylabel('Object Count')
    plt.title('Object Count per Class')
    plt.xticks(rotation=90)

    for i in range(len(classes)):
        plt.text(i, counts[i], str(counts[i]), ha='center', va='bottom', fontsize=8, rotation=90)

    # 根据objects个数，手动设置Y轴刻度
    plt.yticks([i * 2000 for i in range((max(counts) // 2000) + 2)])

    plt.tight_layout()
    plt.show()


# 示例用法：
directory = r'I:\match\match3\train_data\labelTxt_QZB_extracte_new'
class_counts = count_objects_in_txt_files(directory)
plot_bar_chart(class_counts)
