import os
import torch
import sys
import torch.nn as nn
import numpy as np

# 添加 system 目录到 PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./system"))
sys.path.append(project_path)

# 现在可以正常导入 flcore
import flcore

import inspect

def print_caller_info():
    # 获取调用栈中的上一帧信息
    caller = inspect.stack()[1]
    print(f"=====HPDIC DEBUG=====\nCalled from file: {caller.filename}, line: {caller.lineno}, function: {caller.function}")

def print_structure(data, indent=0):
    spacing = "  " * indent
    if isinstance(data, dict):
        print(f"{spacing}(Dict) Keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"{spacing}  {key}: ", end="")
            print_structure(value, indent + 1)
    elif isinstance(data, list):
        print(f"{spacing}(List) Length: {len(data)}")
        for i, item in enumerate(data[:5]):  # 限制打印前 5 项
            print(f"{spacing}  [{i}]: ", end="")
            print_structure(item, indent + 1)
    elif isinstance(data, torch.Tensor):
        print(f"{spacing}(Tensor) Shape: {data.shape}, Dtype: {data.dtype}")
    else:
        print(f"{spacing}{type(data)}: {data}")

def extract_flattened_weights(pt_file_path, output_dir="results"):
    """
    从 .pt 文件中提取所有权重，并将它们展平为一个大的 NumPy 数组。

    Args:
        pt_file_path (str): .pt 文件路径。

    Returns:
        numpy_array (np.ndarray): 包含所有权重的 NumPy 数组。
    """
    try:
        # 加载 .pt 文件
        data = torch.load(pt_file_path, map_location="cpu", weights_only=False)
        
        # 检查是否是 state_dict 类型
        if hasattr(data, "state_dict"):  # 如果保存的是整个模型
            state_dict = data.state_dict()
        elif isinstance(data, dict):  # 如果直接保存的是 state_dict
            state_dict = data
        else:
            raise ValueError("Unsupported .pt file structure.")

        # 提取并展平所有权重
        weights_list = []
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):  # 只处理张量类型
                weights_list.append(tensor.flatten().numpy())  # 转为 NumPy 数组并展平

        # 合并为一个大数组
        flattened_weights = np.concatenate(weights_list)

        # Prepare output file path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base_name = os.path.splitext(os.path.basename(os.path.dirname(pt_file_path)))[0]
        output_file_path = os.path.join(output_dir, f"numpy_{base_name}.npy")

        # Save to file
        np.save(output_file_path, flattened_weights)
        print(f"Numpy array saved to: {output_file_path}")        

        print(f"Extracted flattened weights with shape: {flattened_weights.shape}")
        return flattened_weights

    except Exception as e:
        print(f"Error extracting weights: {e}")
        return None

if __name__ == "__main__":

    # Structure
    if len(sys.argv) < 2:
        print("Usage: python ptdump.py <path_to_pt_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        print(f"Contents of {file_path}:")
        print_structure(data)
    except Exception as e:
        print(f"Error loading file: {e}")

    # Number of parameters
    # initialize_and_count_params()

    # Load model as an array of numbers
    flattened_weights = extract_flattened_weights(file_path)
    # 打印前 3 和后 3 个值
    print("First 3 values:", flattened_weights[:3])
    print("Last 3 values:", flattened_weights[-3:])

