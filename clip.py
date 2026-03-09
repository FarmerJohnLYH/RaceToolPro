from PIL import Image

# 加载图像
input_file = "v1video_frames/2026-01-31_15-47-51_Front_frame_004912.jpg"
img = Image.open(input_file)
width, height = img.size
print("Original image size:", width, height)
# 定义裁切区域 [左, 上, 右, 下]
# 根据观察，赛道图大约在右下角 20% 的宽度和 30% 的高度范围内
left = 1570
top = height * 0.7    # 从顶部 72% 处开始
right = width  # 结束于右侧 98% 处
bottom = height # 结束于底部 96% 处

# 执行裁切
track_map = img.crop((left, top, right, bottom))
print("Size = ", track_map.size)
# 保存结果
track_map.save("track_map_crop.png")
print("裁切完成，已保存为 track_map_crop.png")
