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
        "刹车力度": [[np.int32(181), np.int32(1609)], [np.int32(290), np.int32(1609)], [np.int32(290), np.int32(1673)], [np.int32(181), np.int32(1673)]],
        "油门力度": None,
        "LapTime": [[np.int32(1595), np.int32(1634)], [np.int32(2244), np.int32(1634)], [np.int32(2244), np.int32(1751)], [np.int32(1595), np.int32(1751)]],
        "电机功率": None,
        "时速": [[np.int32(959), np.int32(1775)], [np.int32(1200), np.int32(1775)], [np.int32(1200), np.int32(1876)], [np.int32(959), np.int32(1876)]]
    }
def extract_text_with_easyocr(image, output_dir, file_prefix):
    """
    使用EasyOCR提取图像中的文字
    """
    print("正在使用EasyOCR识别文字...")
    # 初始化OCR读取器 - 使用中文和英文

    # 执行OCR识别
    results = reader.readtext(image,
                            allowlist='0123456789.:')

    # 提取识别结果
    extracted_text = []
    for (bbox, text, prob) in results:
        # 只保留置信度较高的结果
        if prob > 0.5:
            extracted_text.append((text, prob, bbox))
            print(f"识别文字: {text} (置信度: {prob:.6f})")

    # 在图像上标记识别区域
    annotated_img = image.copy()
    for (text, prob, bbox) in extracted_text:
        # 绘制边界框
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_img, [pts], True, (0, 255, 0), 2)
        # 添加文字标签
        cv2.putText(annotated_img, text, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 保存标注后的图像
    annotated_path = os.path.join(output_dir, f"{file_prefix}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated_img)
    print(f"标注后的图像已保存至: {annotated_path}")

    return extracted_text, annotated_img
def crop(image, area):
    """
    根据给定的矩形区域裁剪图像
    """
    x1 = min(area[0][0], area[1][0], area[2][0], area[3][0])
    y1 = min(area[0][1], area[1][1], area[2][1], area[3][1])
    x2 = max(area[0][0], area[1][0], area[2][0], area[3][0])
    y2 = max(area[0][1], area[1][1], area[2][1], area[3][1])
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img
def extract_infos(text_results, image):
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
    var = 5  # 允许区间误差
    for text, prob, bbox in text_results:
        # 查找区域 珠海国际赛车场"[[np.int32(3126), np.int32(64)], [np.int32(3490), np.int32(64)], [np.int32(3490), np.int32(130)], [np.int32(3126), np.int32(130)]]"
        for key, area in dict.items():
            if(area is not None):
                if all(abs(bbox[i][0] - area[i][0]) <= var and abs(bbox[i][1] - area[i][1]) <= var for i in range(4)):
                    results[key] = text
                    print(f"识别到 {key}: {text} (置信度: {prob:.6f})")
    for key, value in dict.items():
        if(value is not None):
            add_results = reader.readtext(crop(image, value),
                                allowlist='0123456789.:')
            print("add_results=", add_results)
            for (bbox, text, prob) in add_results:
                results[key] = text
                print(f"在 {key} 区域识别到: {text} (置信度: {prob:.6f})")
    return results

def process_badge(image_path, index=0):
    """
    处理图片，识别名字、工号，并截取头像
    """
    # 记录开始处理时间
    start_time = time.time()
    process_datetime = datetime.datetime.now()
    process_minute = process_datetime.minute
    process_second = process_datetime.second

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：文件 {image_path} 不存在")
        return None

    # 读取图像
    print(f"正在读取图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法读取图像")
        return None

    # 获取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(image_path))[0]

    # 创建结果目录
    output_dir = os.path.join(os.path.dirname(image_path), "ocr_results")
    os.makedirs(output_dir, exist_ok=True)

    # 按照格式重命名结果图像：序号_输入文件名_分钟_秒
    file_prefix = f"{index}_{input_filename}_{process_minute}_{process_second}"

    # 保存原始图像副本
    original_copy_path = os.path.join(output_dir, f"{file_prefix}_original.jpg")
    cv2.imwrite(original_copy_path, image)
    print(f"原始图像已保存至: {original_copy_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_dir, f"{file_prefix}_gray.jpg")
    cv2.imwrite(gray_path, gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    thresh_path = os.path.join(output_dir, f"{file_prefix}_preprocessed.jpg")
    cv2.imwrite(thresh_path, thresh)
    print(f"预处理图像已保存至: {thresh_path}")
    text_results, annotated_img = extract_text_with_easyocr(image, output_dir, file_prefix)

    # 将text_results 导出为表格 (text, prob, bbox))
    text_results_path = os.path.join(output_dir, f"{file_prefix}_text_results.csv")
    df_text_results = pd.DataFrame(text_results, columns=["text", "probability",
                                                        "bbox"])
    df_text_results.to_csv(text_results_path, index=False, encoding='utf-8-sig')
    print(f"OCR识别结果已保存至: {text_results_path}")

    # flag1
    infos = extract_infos(text_results, image)

    end_time = time.time()
    process_time = end_time - start_time
    return None

# 主函数
def main():
    input_dir = "./input"

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在")
        # 创建输入目录
        os.makedirs(input_dir, exist_ok=True)
        print(f"已创建输入目录: {input_dir}")
        print(f"请将图片放入该目录后重新运行程序")
        return

    # 获取目录中的所有图片文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)


    if not image_files:
        print(f"错误：在 {input_dir} 目录中未找到图片文件")
        print(f"请将图片放入该目录后重新运行程序")
        return

    print(f"找到 {len(image_files)} 个图片文件，开始处理...")

    # 创建结果表格数据
    results_data = []

    # 处理每个图片
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 处理图片: {os.path.basename(image_path)}")

        result = process_badge(image_path, i+1)
        if not result:
            print(f"处理失败: {image_path}")
            continue

        print("\n处理完成！结果摘要:")
        print("-" * 50)

        if result["name"]:
            print(f"识别到的姓名: {result['name']}")
        else:
            print("未能识别姓名")

        if result["employee_id"]:
            print(f"识别到的工号: {result['employee_id']}")
        else:
            print("未能识别工号")

        print(f"工牌区域已保存至: {result['badge_path']}")
        print(f"所有处理结果保存在: {result['output_dir']}")
        print(f"处理时间: {result['process_time']:.2f} 秒")
        print("-" * 50)

        # 显示所有识别到的文本
        print("\n所有识别到的文本:")
        for text, prob, _ in result["text_results"]:
            print(f"- {text} (置信度: {prob:.6f})")

        # 添加结果到表格数据
        row_data = {
            "序号": i+1,
            "文件名": os.path.basename(image_path),
            "姓名": result["name"] if result["name"] else "未识别",
            "工号": result["employee_id"] if result["employee_id"] else "未识别",
            "处理时间(秒)": f"{result['process_time']:.2f}",
            "处理分钟": result["process_minute"],
            "处理秒": result["process_second"],
            "工牌图像路径": result["badge_path"]
        }

        # 添加所有识别文本和置信度
        for j, (text, prob, _) in enumerate(result["text_results"]):
            row_data[f"文本{j+1}"] = text
            row_data[f"置信度{j+1}"] = f"{prob:.6f}"

        results_data.append(row_data)

    # 将结果保存为CSV表格
    if results_data:
        # 创建DataFrame
        df = pd.DataFrame(results_data)

        # 保存为CSV文件
        csv_path = os.path.join(input_dir, "ocr_results", "recognition_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n识别结果已保存到表格: {csv_path}")

if __name__ == "__main__":
    main()