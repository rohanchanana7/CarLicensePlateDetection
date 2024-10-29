import os
import shutil

def save_yolo_format(row, labels_path, images_path):
    x_center = (row['xmin'] + row['xmax']) / 2 / row['img_w']
    y_center = (row['ymin'] + row['ymax']) / 2 / row['img_h']
    width = (row['xmax'] - row['xmin']) / row['img_w']
    height = (row['ymax'] - row['ymin']) / row['img_h']

    label_path = os.path.join(labels_path, f"{os.path.splitext(os.path.basename(row['img_path']))[0]}.txt")
    with open(label_path, 'w') as file:
        file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")
    shutil.copy(row['img_path'], images_path)

def make_split_folder_in_yolo_format(split_name, split_df):
    labels_path = os.path.join('datasets', 'cars_license_plate_new', split_name, 'labels')
    images_path = os.path.join('datasets', 'cars_license_plate_new', split_name, 'images')
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    for _, row in split_df.iterrows():
        save_yolo_format(row, labels_path, images_path)
