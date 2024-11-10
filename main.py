import gradio as gr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from functools import lru_cache
from processed_data.data_processor import preprocess_movielens_data


# 推荐系统核心逻辑
def get_recommendations(user_id, n_recommendations=5):
    """获取电影推荐结果.

    基于协同过滤的推荐算法, 计算用户之间的相似度, 为目标用户推荐电影.

    Args:
        user_id: 用户ID
        n_recommendations: 推荐电影数量, 默认为5

    Returns:
        包含推荐电影信息的字典列表, 每个字典包含电影标题和类型
    """
    movies, user_movie_matrix, user_to_index, movie_to_index = load_movielens_data()

    # 检查用户是否存在
    if user_id not in user_to_index:
        return []

    user_idx = user_to_index[user_id]
    user_vector = user_movie_matrix[user_idx].toarray().flatten()
    rated_items = user_vector.nonzero()[0]

    if len(rated_items) == 0:
        return []

    # 计算用户评分过的电影与所有电影的相似度
    similarities = cosine_similarity(
        user_movie_matrix[:, rated_items].T,
        user_movie_matrix.T
    )

    # 基于相似度计算预测评分
    pred_ratings = np.zeros(user_movie_matrix.shape[1])
    for i, item_idx in enumerate(rated_items):
        pred_ratings += similarities[i] * user_vector[item_idx]

    # 排除已看过的电影
    pred_ratings[rated_items] = float('-inf')
    movie_indices = np.argsort(pred_ratings)[::-1][:n_recommendations]

    # 获取推荐电影信息
    index_to_movie = {v: k for k, v in movie_to_index.items()}
    recommended_movies = [index_to_movie[idx] for idx in movie_indices]
    recommendations = movies[movies['movieId'].isin(recommended_movies)]
    
    return recommendations[['title', 'genres']].to_dict('records')


# 推荐结果展示函数
def recommend_movies(user_id):
    """生成电影推荐展示文本.

    Args:
        user_id: 用户ID

    Returns:
        str: 包含推荐电影信息的格式化文本
    """
    try:
        user_id = int(user_id)
        recommendations = get_recommendations(user_id)

        if not recommendations:
            return "未找到该用户或无法生成推荐"

        output = "推荐电影列表:\n\n"
        for i, rec in enumerate(recommendations, 1):
            output += f"{i}. 电影名称: {rec['title']}\n"
            output += f"   类型: {rec['genres']}\n\n"

        return output

    except ValueError:
        return "请输入有效的用户ID"


# Gradio界面配置
with gr.Blocks(title="电影推荐系统", theme=gr.themes.Soft()) as recSys:
    gr.Markdown("# 简单电影推荐系统")

    with gr.Row():
        with gr.Column():
            user_input = gr.Number(
                label="用户ID",
                value=1,
                minimum=1,
                step=1
            )

    with gr.Row():
        submit_btn = gr.Button("获取推荐")

    with gr.Row():
        output = gr.Textbox(
            label="推荐结果",
            lines=10
        )

    submit_btn.click(
        fn=recommend_movies,
        inputs=[user_input],
        outputs=[output]
    )


# 数据加载函数
@lru_cache(maxsize=1)
def load_movielens_data():
    """加载预处理的数据, 如果不存在则进行预处理.

    使用lru_cache装饰器缓存结果, 避免重复加载.

    Returns:
        tuple: (movies, user_movie_matrix, user_to_index, movie_to_index)
    """
    processed_data_path = 'processed_data/movielens_data.pkl'

    if os.path.exists(processed_data_path):
        print("从本地加载预处理数据...")
        with open(processed_data_path, 'rb') as f:
            data = pickle.load(f)
        print("数据加载完成")
    else:
        print("未找到预处理数据, 开始预处理...")
        data = preprocess_movielens_data()

    return (
        data['movies'],
        data['user_movie_matrix'],
        data['user_to_index'],
        data['movie_to_index']
    )


def force_preprocess():
    """强制重新预处理数据."""
    return preprocess_movielens_data()


def clear_data_cache():
    """清除数据缓存."""
    load_movielens_data.cache_clear()


# 主程序入口
if __name__ == "__main__":
    # force_preprocess()
    recSys.launch()
