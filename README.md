# SPEECHHOME-AISHELL-3

> 基于 Tacotron-2 实现的 AISHELL-3 多说话人语音合成实验


## 1. 目录结构
```
root
    configs/                        # 超参数配置文件
    melgan/                         # MelGAN 模型实现
        hubconf.py                  # MelGAN 接口
        ...
    model/                          # Tacotron-2 模型实现
        main.py                     # Tacotron-2 训练控制
        model.py                    # Tacotron-2 实现
        hybrid_attention.py         # 混合注意力机制实现
        gmm_attention.py            # GMM 注意力机制实现
    text/                           # 文本预处理实现
    audio_processing.py             # 语音预处理工具
    data_utils.py                   # 数据集实现
    plot.py                         # 画图相关
    preprocess_ai3.py               # aishell-3 数据预处理
    train.py                        # 模型训练入口脚本
    vaildate_dataset.py             # 数据集测试
    inference.ipynb                 # 模型测试
    env.yaml                        # conda 依赖说明
    README.py                       # 说明文件
```


## Acknowledgement

The Tacotron model itself was adapted from https://github.com/NVIDIA/tacotron2.
`melgan` is a fork from https://github.com/seungwonpark/melgan.

