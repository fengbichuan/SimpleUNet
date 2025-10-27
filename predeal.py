
import os
import glob

# --- 请修改这里的设置 ---

# 1. 目标文件夹路径
#    '.' 代表当前脚本所在的文件夹
#    您也可以指定一个绝对路径, 例如: r'D:\MyPhotos'

folder_path = r'D:\对照试验模型\dataset\8-isic2017\val\images'


# 2. 定义哪些文件被视为“照片”
#    脚本将重命名所有以这些扩展名结尾的文件
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

# 3. 定义要删除的文件模式
delete_pattern = '*_superpixels.png'


# --- 脚本正文 ---

def delete_files(path, pattern):
    """
    第一步：删除匹配特定模式的文件
    """
    print(f"--- 步骤 1: 开始删除文件 ---")
    file_pattern = os.path.join(path, pattern)
    files_to_delete = glob.glob(file_pattern)

    if not files_to_delete:
        print(f"在 '{path}' 中没有找到匹配 '{pattern}' 的文件。")
        print("------------------------------\n")
        return

    print(f"找到了以下匹配 '{pattern}' 的文件，即将删除：")
    for f in files_to_delete:
        print(f"  - {f}")

    confirm = input("您确定要删除以上所有文件吗？ (输入 'y' 确认): ")

    if confirm.lower() == 'y':
        deleted_count = 0
        failed_count = 0
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as e:
                print(f"删除失败: {file_path} (错误: {e})")
                failed_count += 1

        print(f"\n操作完成。成功删除 {deleted_count} 个文件，失败 {failed_count} 个。")
    else:
        print("删除操作已取消。")

    print("------------------------------\n")


def rename_files(path, extensions, exclude_pattern):
    """
    第二步：重命名文件夹中剩余的图片
    """
    print(f"--- 步骤 2: 开始重命名剩余图片 ---")

    # 1. 收集所有需要重命名的文件
    all_files = os.listdir(path)
    files_to_rename = []
    for f in all_files:
        # 必须是定义的图片扩展名
        if f.lower().endswith(extensions):
            # 并且不能是即将被删除的模式 (以防万一删除步骤被跳过)
            if not f.endswith(exclude_pattern):
                files_to_rename.append(f)

    # 2. 按字母顺序排序，以确保重命名顺序一致
    files_to_rename.sort()

    if not files_to_rename:
        print("没有找到需要重命名的照片文件。")
        print("------------------------------\n")
        return

    print(f"即将按以下顺序重命名 {len(files_to_rename)} 个文件：")
    count = 0
    for f in files_to_rename:
        ext = os.path.splitext(f)[1]
        print(f"  - {f}  ->  {count}{ext}")
        count += 1

    confirm = input("您确定要开始重命名吗？ (输入 'y' 确认): ")
    if confirm.lower() != 'y':
        print("重命名操作已取消。")
        print("------------------------------\n")
        return

    # 3. 两阶段重命名 (防止冲突)
    #    阶段 1: a.jpg -> 0.jpg_temp
    temp_suffix = "_temp_rename"
    rename_map = {}  # 存储 {旧文件名: (临时路径, 最终路径)}

    count = 0
    for old_name in files_to_rename:
        ext = os.path.splitext(old_name)[1].lower()
        new_name = f"{count}{ext}"
        temp_name = f"{count}{ext}{temp_suffix}"

        old_path = os.path.join(path, old_name)
        temp_path = os.path.join(path, temp_name)
        final_path = os.path.join(path, new_name)

        rename_map[old_name] = (temp_path, final_path)

        # 执行阶段 1 重命名
        try:
            os.rename(old_path, temp_path)
            print(f"阶段1: {old_name} -> {temp_name}")
        except Exception as e:
            print(f"错误：重命名 {old_name} 到 {temp_name} 失败: {e}")
            print("操作已终止，请检查文件夹。")
            return

        count += 1

    #    阶段 2: 0.jpg_temp -> 0.jpg
    print("\n--- 阶段 2: 移除临时后缀 ---")
    for old_name in files_to_rename:
        temp_path, final_path = rename_map[old_name]
        try:
            os.rename(temp_path, final_path)
            print(f"阶段2: {os.path.basename(temp_path)} -> {os.path.basename(final_path)}")
        except Exception as e:
            print(f"错误：重命名 {temp_path} 到 {final_path} 失败: {e}")
            print("请检查文件夹。")
            return

    print(f"\n重命名操作全部完成。总共重命名了 {count} 个文件。")
    print("------------------------------\n")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 确保路径存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹路径 '{folder_path}' 不存在。")
    else:
        # 第一步：删除
        delete_files(folder_path, delete_pattern)

        # 第二步：重命名
        rename_files(folder_path, image_extensions, delete_pattern)