import glob
import os

# 修改为你的 DIC-C2DH-HeLa 根目录绝对路径
data_root = "/home/monkey/下载/U-net数据集/DIC-C2DH-HeLa_train/DIC-C2DH-HeLa"

pairs = []

# 遍历 01 和 02 两个子序列
for seq in ["01", "02"]:
    seg_dir = os.path.join(data_root, f"{seq}_GT", "SEG")
    # 找到所有的 man_seg*.tif
    seg_files = sorted(glob.glob(os.path.join(seg_dir, "man_seg*.tif")))

    for seg_path in seg_files:
        filename = os.path.basename(seg_path)  # 如 man_seg002.tif
        num_str = filename.replace("man_seg", "").replace(".tif", "")  # 取出 002
        img_filename = f"t{num_str}.tif"  # 拼接出原图名字 t002.tif
        img_path = os.path.join(data_root, seq, img_filename)

        if os.path.exists(img_path):
            pairs.append((img_path, seg_path))

print(f"成功匹配到的样本总数: {len(pairs)}")
if len(pairs) > 0:
    print("第一组样本示例:")
    print("原图:", pairs[0][0])
    print("标签:", pairs[0][1])