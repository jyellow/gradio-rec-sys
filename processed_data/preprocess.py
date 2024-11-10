import os
import sys
from processed_data.data_processor import preprocess_movielens_data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def main():
    """
    运行数据预处理脚本
    """
    try:
        print("开始运行数据预处理...")
        processed_data = preprocess_movielens_data()
        print("数据预处理成功完成!")
        print(f"处理后的数据包含 {processed_data['user_movie_matrix'].shape[0]} 个用户")
        print(f"和 {processed_data['user_movie_matrix'].shape[1]} 部电影")
    except Exception as e:
        print(f"数据预处理过程中发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
