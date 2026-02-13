import cv2
import numpy as np
import easyocr
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法读取图像")
        return None
    
    # 针对 1920*1080 的特定区域进行裁剪识别
    infos = extract_infos(image) 
    return infos

import re

def laptime_to_seconds(laptime_str):
    if pd.isna(laptime_str) or laptime_str is None:
        return 0.0
    # 匹配 分:秒.毫秒 ,  分.秒.毫秒, 秒.毫秒 格式
    match = re.match(r'(\d+):(\d+\.\d+)', str(laptime_str))
    if match:
        return int(match.group(1))*60 + float(match.group(2))
    
    match = re.match(r'(\d+)\.(\d+\.\d+)', str(laptime_str))
    if match:
        return int(match.group(1))*60 + float(match.group(2))
    match2 = re.match(r'(\d+\.\d+)', str(laptime_str))
    if match2:
        return float(match2.group(1))
    return 0.0

def clean_speed_value(val):
    if pd.isna(val) or val is None:
        return 0.0
        # 提取数字和小数点，过滤其他字符
    num_str = re.findall(r'[\d\.]+', str(val))
    if num_str:
        try:
            return float(''.join(num_str))
        except:
            return 0.0
    return 0.0
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
    skipped_frame = 10  # 每 30 帧跑一次. 最后改成 1
    for i, image_path in enumerate(tqdm(image_files)): 
        if(i < 4841 or i > 8350): # 仅针对 2026-01-31_15-47-51_Front
            continue 
        if(i%skipped_frame != 0):
            continue
        result = process_one_frame(image_path, i+1)
        if (not result) or (result["LapTime"] is None):
            print(f"处理失败: {image_path}")
            continue
        result["image_name"] = os.path.basename(image_path)
        results_data.append(result)
       
        stream_save = False # 实时保存到 csv 表格 
        if (stream_save): 
            df = pd.DataFrame([result])
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            else:
                df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')

    df_result = pd.DataFrame(results_data)
    df_result.to_csv(csv_path[:-4] + "_final.csv", index=False, encoding='utf-8-sig') # 保存最终结果到 csv 文件
    # 计算时速的极值点，并绘图 

    se = df_result.copy().reset_index(drop=True)  # 重置索引为连续有序的0,1,2...
    se['Speed_value'] = se['时速'].apply(clean_speed_value)
    se['LapTime_sec'] = se['LapTime'].apply(laptime_to_seconds)


    peak_indices, _ = find_peaks(se['Speed_value'], height=0, distance=5)  # distance避免相邻帧重复极值
    valley_indices, _ = find_peaks(-se['Speed_value'], distance=5)
    peak_indices = np.concatenate([peak_indices, valley_indices])


    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 解决中文显示
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(se['LapTime_sec'], se['Speed_value'], 
            color='blue', linewidth=3, label='时速曲线')
    ax.scatter(se['LapTime_sec'].iloc[peak_indices], se['Speed_value'].iloc[peak_indices],
               color='red', s=50, zorder=5)

    for idx in peak_indices:
        x = se['LapTime_sec'].iloc[idx]
        y = se['Speed_value'].iloc[idx]
        ax.text(x + 0.5, y + 1,
                f'({x:.1f}s, {y:.0f}km/h)',
                fontsize=8, color='red', ha='left', va='bottom')

    xticks = np.arange(0, np.ceil(se['LapTime_sec'].max() / 10) * 10 + 10, 10)             # 生成10秒间隔的刻度（如0,10,20...）
    ax.set_xticks(xticks)
    yticks = np.arange(0, np.ceil(se['Speed_value'].max() / 25) * 25+25, 25)             # 生成25km/h间隔的刻度（如0,25,50...）
    ax.set_yticks(yticks)
    plt.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.8)
    plt.tight_layout()
    plot_save_path = os.path.join(input_dir, "speed_extremes_plot.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()