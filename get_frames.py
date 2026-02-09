import cv2
import os

def extract_frames_from_video(video_path, output_dir="output_frames", frame_prefix=None):
    """
    从视频中提取每一帧并保存为图片
    :param video_path: 输入视频路径（如 input/2026-01-31_15-47-51_Front.mp4）
    :param output_dir: 输出图片的目录（默认 output_frames）
    :param frame_prefix: 图片名前缀（默认用视频文件名）
    :return: dict: 包含提取结果、总帧数、错误信息的字典
    """
    # 1. 校验输入视频是否存在
    if not os.path.exists(video_path):
        return {"success": False, "error": f"视频文件不存在：{video_path}", "total_frames": 0}
    
    # 2. 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 提取视频文件名作为图片前缀（去掉扩展名）
    if frame_prefix is None:
        video_name = os.path.basename(video_path)
        frame_prefix = os.path.splitext(video_name)[0]  # 得到 2026-01-31_15-47-51_Front
    
    # 4. 打开视频流
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "无法打开视频文件（格式不支持或文件损坏）", "total_frames": 0}
    
    # 5. 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率（可选，用于参考）
    print(f"开始提取帧：\n- 视频路径：{video_path}\n- 总帧数：{total_frames}\n- 帧率：{fps:.2f} FPS")
    
    # 6. 逐帧读取并保存
    frame_count = 0
    success = True
    
    while success:
        # 读取一帧
        success, frame = cap.read()
        
        if success:
            # 定义图片保存路径（格式：输出目录/前缀_帧序号.jpg）
            frame_filename = f"{frame_prefix}_frame_{frame_count:06d}.jpg"  # 06d 补零到6位，保证排序正确
            save_path = os.path.join(output_dir, frame_filename)
            
            # 保存图片（质量默认95，可调整）
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 进度提示（每100帧打印一次）
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"进度：{frame_count}/{total_frames} 帧 ({progress:.1f}%)")
            
            frame_count += 1
    
    # 7. 释放资源
    cap.release()
    
    # 8. 结果校验
    if frame_count == 0:
        return {"success": False, "error": "未提取到任何帧（视频为空或读取失败）", "total_frames": 0}
    else:
        print(f"\n提取完成！共保存 {frame_count} 帧图片，输出目录：{os.path.abspath(output_dir)}")
        return {
            "success": True,
            "error": "",
            "total_frames": frame_count,
            "output_dir": os.path.abspath(output_dir)
        }

# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 替换为你的视频路径
    VIDEO_PATH = "input/2026-01-31_15-47-51_Front.mp4"
    # 可选：自定义输出目录
    OUTPUT_DIR = "video_frames"
    
    # 执行帧提取
    result = extract_frames_from_video(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # 输出结果
    if not result["success"]:
        print(f"❌ 提取失败：{result['error']}")
    else:
        print(f"✅ 提取成功！共 {result['total_frames']} 帧，保存至：{result['output_dir']}")