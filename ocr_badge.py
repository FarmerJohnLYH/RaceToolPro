import cv2
import numpy as np
import easyocr
import os
import re
import time
import datetime
import pandas as pd
from PIL import Image

reader = easyocr.Reader(['en'])
dict = {
        "刹车力度": [[np.int32(85), np.int32(799)], [np.int32(151), np.int32(799)], [np.int32(151), np.int32(843)], [np.int32(85), np.int32(843)]],
        "油门力度": [[np.int32(216), np.int32(806)], [np.int32(272), np.int32(806)], [np.int32(272), np.int32(838)], [np.int32(216), np.int32(838)]],
        "LapTime": [[np.int32(788), np.int32(810)], [np.int32(1134), np.int32(810)], [np.int32(1134), np.int32(882)], [np.int32(788), np.int32(882)]],
        "电机功率": [[np.int32(1012), np.int32(980)], [np.int32(1096), np.int32(980)], [np.int32(1096), np.int32(1028)], [np.int32(1012), np.int32(1028)]],
        "时速": [[np.int32(492), np.int32(888)], [np.int32(576), np.int32(888)], [np.int32(576), np.int32(936)], [np.int32(492), np.int32(936)]],
    }
def extract_text_with_easyocr(image, output_dir, file_prefix):
    """
    使用EasyOCR提取图像中的文字
    """
    print("正在使用EasyOCR识别文字...")
    # 方案 1 全图识别
    results = reader.readtext(image,
                            allowlist='0123456789.:')
    annotated_img = image.copy()
    for (bbox, text, prob) in results:
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_img, [pts], True, (0, 255, 0), 2)
        cv2.putText(annotated_img, text, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    annotated_path = os.path.join(output_dir, f"{file_prefix}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated_img)
    print(f"标注后的图像已保存至: {annotated_path}")
    return results, annotated_img
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
    input_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(os.path.dirname(image_path), "ocr_results")
    os.makedirs(output_dir, exist_ok=True)
    # 按照格式重命名结果图像：序号_输入文件名
    file_prefix = f"{index}_{input_filename}_"
    # text_results, annotated_img = extract_text_with_easyocr(image, output_dir, file_prefix)
        # text_results_path = os.path.join(output_dir, f"{file_prefix}_text_results.csv")
        # df_text_results = pd.DataFrame(text_results, columns=["bbox", "text", "probability"])
        # df_text_results.to_csv(text_results_path, index=False, encoding='utf-8-sig')
        # print(f"OCR识别结果已保存至: {text_results_path}")
    # 方案 2 针对 1920*1080 的特定区域进行裁剪识别
    infos = extract_infos(image) 
    return infos

def main():
    input_dir = "./input"
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
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 处理图片: {os.path.basename(image_path)}")
        result = process_one_frame(image_path, i+1)
        if not result:
            print(f"处理失败: {image_path}")
            continue
        result["image_name"] = os.path.basename(image_path)
        results_data.append(result)
    # 将结果保存为CSV表格
    if results_data:
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(input_dir, "ocr_results", "recognition_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n识别结果已保存到表格: {csv_path}")

if __name__ == "__main__":
    main()