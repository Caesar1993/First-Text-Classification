# -*- coding: utf-8 -*-
import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

'''
同类模型不用做太多重复，lstm和gru挑一个
样本少时，使用bert模型。
1-在模型搭建完后，训练集和测试集使用同一组数据，如果拟合，准确率达到100%，则表明模型设计没有大问题。
2-使用12层bert能达到87%
'''
# [DEBUG,INFO,WARNING,ERROR,CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 日志模块的吸心大法，__name__是当前py的子用户，都是root下边的，每个py设置自己的子用户，打印等级也在自己的范围内生效
# 所有想用print的地方，都改成logging.info

"""
模型训练主程序
"""
# 调优的时候要固定随机数，减少随机数的干扰
# 训练阶段，在config文件中，设置不同的seed，训练结果都差不多，就说明不是局部最优
# 对于机器来说，没有理论上的随机数，用非常散列的算法生成随机数

seed = Config["seed"]
# 四个随机种子
random.seed(seed)  # 在加载数据的模块用到。把训练数据做一次洗牌。保证多次调用生成的随机数相同
np.random.seed(seed)
torch.manual_seed(seed)  # 为cpu设置种子随机数
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子随机数


# return_dict=False


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)  # 实例化模型，执行TorchModel的init方法
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()  # 用torch的此方法，明确能否使用gpu
    # cuda_flag = False# 不用gpu
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()  # 调用gpu的步骤一，将模型参数放到gpu上
    # 加载优化器。反向传播时修改权重基于的逻辑
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):  # 一共1440个训练样本，每批64个，所以分为23批，最后一批不满
            if cuda_flag:  # 如果使用gpu，则把训练数据也放到gpu上
                batch_data = [d.cuda() for d in batch_data]
                # batch_data = (d.cuda() for d in batch_data)#这是生成器
                # batch_data = batch_data.cuda()#不行

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())  # tensor.item，将一个Tensor转换为一个Python number
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)  # 把这个东西传给了message
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)  # 训练一次进行一次评估

    model_path = os.path.join(config["model_path"],
                              "%s-epoch_%d.pth" % (config["model_type"], epoch))  # 应该加入model_type，保存模型
    # model_path = os.path.join(config["model_path"],"epoch_%d.pth" %epoch)  # 应该加入model_type，保存模型
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc


if __name__ == "__main__":
    # main(Config)

    # # for model in ['cnn','gated_cnn','stack_gated_cnn']:
    # for model in ['bert_lstm']:
    #  for model in ['cnn']:
     for model in ['lstm']:
    # for model in ['bert']:
        Config['model_type'] = model
        print('最后一轮准确率:', main(Config), '当前配置:', Config['model_type'])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索,也可以第三方的超参数库
    # for model in ["gated_cnn","lstm","fast_text"]:
    #     Config["model_type"] = model
    #     for lr in [1e-2, 1e-3]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [64, 128, 256]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["max", "avg"]:
    #                     Config["pooling_style"] = pooling_style
    #                     logger.info("最后一轮准确率：", main(Config), "当前配置：", Config)
