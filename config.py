# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../train_tag_news.json",
    "valid_data_path": "../valid_tag_news.json",
    "vocab_path": "../chars.txt",
    "model_type": "gated_cnn",  # 默认是这个，但是main程序会更新
    "max_length": 20,
    "hidden_size": 128,  # 给rnn，cnn等设置的，bert的hidden size是预训练的
    "kernel_size": 3,  # 给CNN设置的
    "num_layers": 12,
    "epoch": 20,
    "batch_size": 100,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,  # 训练bert模型学习率取小，1e-4，bert模型越大，学习率越小，1e-5作为起点。使用预训练部分越多，学习率应该越小
    "pretrain_model_path": r"D:\NLP_BaDou_AI\bert-base-chinese",  # 加载bert模型
    "seed": 1024
}
