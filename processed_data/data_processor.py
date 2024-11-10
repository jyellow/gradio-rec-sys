import pandas as pd
import os
import pickle
from scipy.sparse import csr_matrix


def preprocess_movielens_data():
    """预处理 MovieLens 数据并保存到本地.

    处理步骤:
    1. 读取原始数据文件
    2. 构建用户-电影评分矩阵
    3. 创建ID映射关系
    4. 保存处理后的数据

    Returns:
        dict: 包含预处理后数据的字典
    """
    print("开始预处理数据...")

    # 加载原始数据
    ratings = pd.read_csv('../datasets/ml-32m/ratings.csv')
    movies = pd.read_csv('../datasets/ml-32m/movies.csv')

    # 获取唯一用户和电影ID
    users = ratings['userId'].unique()
    movies_list = ratings['movieId'].unique()

    # 创建ID到索引的映射
    user_to_index = {user: idx for idx, user in enumerate(users)}
    movie_to_index = {movie: idx for idx, movie in enumerate(movies_list)}

    # 构建稀疏矩阵的坐标
    row = ratings['userId'].map(user_to_index)
    col = ratings['movieId'].map(movie_to_index)
    data = ratings['rating']

    # 创建用户-电影评分稀疏矩阵
    user_movie_matrix = csr_matrix(
        (data, (row, col)),
        shape=(len(users), len(movies_list))
    )

    processed_data = {
        'movies': movies,
        'user_movie_matrix': user_movie_matrix,
        'user_to_index': user_to_index,
        'movie_to_index': movie_to_index
    }

    # 保存处理后的数据
    os.makedirs('.', exist_ok=True)
    with open('movielens_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)

    print("数据预处理完成, 已保存到 processed_data/movielens_data.pkl")
    return processed_data
