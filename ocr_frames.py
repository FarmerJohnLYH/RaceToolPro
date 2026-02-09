import cv2
import numpy as np
import easyocr
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
reader = easyocr.Reader(['en'])
dict = {
        "刹车力度": [[np.int32(85), np.int32(799)], [np.int32(151), np.int32(799)], [np.int32(151), np.int32(843)], [np.int32(85), np.int32(843)]],
        "油门力度": [[np.int32(216), np.int32(806)], [np.int32(272), np.int32(806)], [np.int32(272), np.int32(838)], [np.int32(216), np.int32(838)]],
        "LapTime": [[np.int32(788), np.int32(810)], [np.int32(1134), np.int32(810)], [np.int32(1134), np.int32(882)], [np.int32(788), np.int32(882)]],
        "电机功率": [[np.int32(1012), np.int32(980)], [np.int32(1096), np.int32(980)], [np.int32(1096), np.int32(1028)], [np.int32(1012), np.int32(1028)]],
        "时速": [[np.int32(492), np.int32(888)], [np.int32(576), np.int32(888)], [np.int32(576), np.int32(936)], [np.int32(492), np.int32(936)]],
}
def crop(image, area):
    var = [10, 10]
    x1 = max(0, min(area[0][0], area[1][0], area[2][0], area[3][0]) - var[0])
    y1 = max(0, min(area[0][1], area[1][1], area[2][1], area[3][1]) - var[1])
    x2 = min(1920, max(area[0][0], area[1][0], area[2][0], area[3][0]) + var[0])
    y2 = min(1080, max(area[0][1], area[1][1], area[2][1], area[3][1]) + var[1])
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img
def extract_infos(image):
    """
    从OCR结果中提取信息。通过bbox定位
    """
    results = {
        "刹车力度": None,
        "油门力度": None,
        "LapTime": None,
        "电机功率": None,
        "时速": None
    }

    for key, value in dict.items():
        if(value is not None):
            # print(f"正在裁剪 {key} 区域进行识别...") #log
            add_results = reader.readtext(crop(image, value),
                                allowlist='0123456789.:',
                                text_threshold=0.5, 
                                low_text=0.2
                                )
            # 按置信度选择最高的 
            if (len(add_results) > 1 or len(add_results) == 0): # log
            #     print(f"警告：在 {key} 区域识别到 {len(add_results)} 个结果，可能存在误识别")
            #     print(f"识别结果: {add_results}")
                pass
            if add_results:
                add_results.sort(key=lambda x: x[2], reverse=True)
                add_results = [add_results[0]]  # 只保留置信度最高的结果
            for (bbox, text, prob) in add_results:
                results[key] = text
    return results

def process_one_frame(image_path, index=0):
    if not os.path.exists(image_path):
        print(f"错误：文件 {image_path} 不存在")
        return None
    print(f"正在读取图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法读取图像")
        return None
    # 方案 2 针对 1920*1080 的特定区域进行裁剪识别
    infos = extract_infos(image) 
    return infos

def main():
    input_dir = "./video_frames"
    # input_dir = "./input"
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在")
        os.makedirs(input_dir, exist_ok=True)
        print(f"已创建输入目录: {input_dir}")
        print(f"请将图片放入该目录后重新运行程序")
        return

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)

    if not image_files:
        print(f"错误：在 {input_dir} 目录中未找到图片文件")
        return
    print(f"找到 {len(image_files)} 个图片文件，开始处理...")
    results_data = []
    csv_path = os.path.join(input_dir, "recognition_results.csv")
    if os.path.exists(csv_path):
        # delete existing csv file to avoid confusion
        os.remove(csv_path)
        print(f"警告：输出文件 {csv_path} 已存在，已删除")
    # 排序 image_files 
    image_files.sort()
    skipped_frame = 30  # 每 30 帧跑一次. 最后改成 1
    for i, image_path in enumerate(tqdm(image_files)): 
        if(i%skipped_frame != 0):
            continue
        result = process_one_frame(image_path, i+1)
        if not result:
            print(f"处理失败: {image_path}")
            continue
        result["image_name"] = os.path.basename(image_path)
        results_data.append(result)
        # 实时保存到 csv 表格 
        df = pd.DataFrame([result])
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()