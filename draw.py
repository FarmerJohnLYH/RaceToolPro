import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

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
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"已创建输入目录: {input_dir}")
    csv_path = "w=3recognition_results_final.csv"
    df_result = pd.read_csv(csv_path)
    se = df_result.copy().reset_index(drop=True)

    print(f"总帧数: {len(se)}")
    se['Speed_value'] = se['时速'].apply(clean_speed_value)
    se['LapTime_sec'] = se['LapTime'].apply(laptime_to_seconds)
    se['color_value'] = -(se['油门力度'] * (se['刹车力度'] > 0).astype(float) - se['刹车力度'])


    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 2. 创建渐变线条 (LineCollection) ---
    x = se['LapTime_sec'].values
    y = se['Speed_value'].values
    c = se['color_value'].values

    # 将点格式化为线段
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 创建 LineCollection，使用 'coolwarm' 映射（蓝-白-红）或 'jet' / 'RdYlBu_r'
    # coolwarm 非常适合这种“正负”逻辑：负数蓝，正数红
    norm = Normalize(vmin=c.min(), vmax=c.max())
    print("whole C = ", c)
    #RdYlBu_r
    lc = LineCollection(segments, cmap='coolwarm', norm=norm) # 基准颜色，一般
    lc = LineCollection(segments, cmap='bwr', norm=norm) # 颜色稍微比 coolwarm 亮一点
    # lc = LineCollection(segments, cmap='seismic', norm=norm) # 看不太出具体变化
    # lc = LineCollection(segments, cmap='RdYlBu_r', norm=norm)
    lc.set_array(c) # 设置颜色映射的数值
    lc.set_linewidth(3)

    # 添加到图表
    ax.plot(se['LapTime_sec'], se['Speed_value'], color='black', linewidth=5, alpha=0.5) # 底色
    line = ax.add_collection(lc)

    peak_distance = 20
    peak_indices, _ = find_peaks(se['Speed_value'], distance=peak_distance)
    valley_indices, _ = find_peaks(-se['Speed_value'], distance=peak_distance)
    peak_indices = np.concatenate([peak_indices, valley_indices])
    ax.scatter(x[peak_indices], y[peak_indices], color='red', s=50, zorder=5)

    for idx in peak_indices:
        ax.text(x[idx] + 1, y[idx] + 1, f'({x[idx]:.1f}s, {y[idx]:.0f}km/h)',
                fontsize=12, color='black', ha='left', va='bottom')

    # --- 4. 坐标轴设置 (强制从 0 开始) ---
    xticks = np.arange(0, np.ceil(x.max() / 10) * 10 + 10, 10)
    yticks = np.arange(0, np.ceil(y.max() / 25) * 25 + 25, 25)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(0, xticks.max())
    ax.set_ylim(0, yticks.max())
    plt.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.8)
    plt.title("赛道时速与操作力度趋势图", fontsize=15)
    plt.xlabel("时间 (s)")
    plt.ylabel("时速 (km/h)")

    plt.tight_layout()

    plot_save_path = os.path.join(input_dir, "赛道时速与操作力度趋势图.png")
    plt.savefig(plot_save_path, dpi=600, bbox_inches='tight')
    print(f"图表已保存至: {plot_save_path}")

if __name__ == "__main__":
    main()
