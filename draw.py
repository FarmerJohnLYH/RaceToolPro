import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import make_interp_spline

import numpy as np
from scipy.interpolate import make_interp_spline

def smooth(x, y, c, num_points, k=3):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x = x[mask].copy()
    y = y[mask].copy()
    c = c[mask].copy()

    unique_mask = np.concatenate([[True], np.diff(x) != 0])
    x = x[unique_mask]
    y = y[unique_mask]
    c = c[unique_mask]

    # spl_y = make_interp_spline(x, y, k=k)
    # y_smooth = spl_y(x_smooth)
    # spl_c = make_interp_spline(x, c, k=k)
    # c_smooth = spl_c(x_smooth)
    return x,y,c

def laptime_to_seconds(laptime_str):
    if pd.isna(laptime_str) or laptime_str is None:
        return 0.0
    # 匹配 分:秒.毫秒 ,  分.秒.毫秒, 秒.毫秒 格式
    match = re.match(r'(\d+):(\d+\.\d+)', str(laptime_str))
    if match:
        return int(match.group(1)) * 60 + float(match.group(2))

    match = re.match(r'(\d+)\.(\d+\.\d+)', str(laptime_str))
    if match:
        return int(match.group(1)) * 60 + float(match.group(2))
    
    match2 = re.match(r'(\d+\.\d+)', str(laptime_str))
    if match2:
        return float(match2.group(1))
    
    return 0.0

def clean_speed_value(val):
    if pd.isna(val) or val is None:
        return 0.0
    num_str = re.findall(r'[\d\.]+', str(val))
    if num_str:
        try:
            return float(''.join(num_str))
        except:
            return 0.0
    return 0.0

def main():
    input_dir = "./v1video_frames"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"已创建输入目录: {input_dir}")

    csv_path = "frame=1recognition_results.csv"
    df_result = pd.read_csv(csv_path)
    se = df_result.copy().reset_index(drop=True)

    print(f"总帧数: {len(se)}")

    se['Speed_value']   = se['时速'].apply(clean_speed_value)
    se['LapTime_sec']   = se['LapTime'].apply(laptime_to_seconds)
    se['Throttle'] = se['油门力度'].apply(clean_speed_value)
    se['Brake'] = se['刹车力度'].apply(clean_speed_value)
    se['color_value']   = -(se['Throttle'] * (se['Brake'] > 0).astype(float) - se['Brake'])

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(15, 8))
    x = se['LapTime_sec'].values
    y = se['Speed_value'].values
    c = se['color_value'].values
    # 将 x,y,c 每三个点取 1 个 
    step = 1
    x = x[::step]
    y = y[::step]
    c = c[::step]

    # x, y, c = smooth(x, y, c, num_points=len(x)*2, k=3)
    # 将点格式化为线段
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 颜色映射：bwr / coolwarm 都适合
    norm = Normalize(vmin=c.min(), vmax=c.max())
    lc = LineCollection(segments, cmap='bwr', norm=norm)
    lc.set_array(c)
    lc.set_linewidth(4)
    lc.set_joinstyle('round')    # 线段连接处平滑圆角
    lc.set_capstyle('round')     # 线段端点平滑圆角
    lc.set_antialiased(True)

    # 黑色底色 + 彩色上层
    # ax.plot(x, y, color='black', linewidth=5, alpha=0.5)
    ax.add_collection(lc)

    # --- 3. 极值点检测 ---
    peak_distance = 20
    peak_indices, _  = find_peaks(y, distance=peak_distance)
    valley_indices, _ = find_peaks(-y, distance=peak_distance)
    extrema_indices = np.concatenate([peak_indices, valley_indices])

    ax.scatter(x[extrema_indices], y[extrema_indices], color='red', s=60, zorder=5)

    for idx in extrema_indices:
        ax.text(x[idx] + 0.5, y[idx] + 2,
                f'({x[idx]:.1f}s, {y[idx]:.0f}km/h)',
                fontsize=10, color='black', ha='left', va='bottom')

    # --- 4. 坐标轴 ---
    if len(x) > 0 and x.max() > 0:
        xticks = np.arange(0, np.ceil(x.max() / 10) * 10 + 10, 10)
    else:
        xticks = [0]
    if len(y) > 0 and y.max() > 0:
        yticks = np.arange(0, np.ceil(y.max() / 25) * 25 + 25, 25)
    else:
        yticks = [0]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(0, xticks.max())
    ax.set_ylim(0, yticks.max())

    plt.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.8)
    plt.title("赛道时速与操作力度趋势图", fontsize=16)
    plt.xlabel("时间 (s)")
    plt.ylabel("时速 (km/h)")

    plt.tight_layout()
    plot_save_path = os.path.join(input_dir, "speed_extremes_plot.png")
    plt.savefig(plot_save_path, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"图表已保存至: {plot_save_path}")

if __name__ == "__main__":
    main()