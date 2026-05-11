# GeoCLIP 优化项目路线图

## 1. 项目目标

该项目是对原始基于 CLIP 的全球地理定位模型（GeoCLIP）的二次开发。原始 GeoCLIP 通过对比学习将街景图像与 GPS 坐标对齐，并在多个开源数据集上展现了良好表现。然而，它在街景数据集上的表现仍有优化空间。

本项目的核心目标在于通过轻量级架构组合及训练策略的改进，提升模型在街景数据集下地理定位的准确性与鲁棒性。

## 2. 实现方案

本次项目的二次开发分为以下三个主要的迭代实施阶段：

### 2.1 引入位置权重的 SigmaSelector
*   **背景**：原始的 `LocationEncoder` 直接以等权重方式相加三种预定义空间频率（粗、中、细粒度 $\sigma_{1,16,256}$）的位置特征 ($f_{loc} = f_0 + f_1 + f_2$)。但在实际应用中，不同地点（如密集的城市中心与开阔的乡村公路）对于不同空间尺度特征的依赖程度必然不同。
*   **实现**：引入轻量级神经网络。该网络接收重投影后的 GPS 坐标作为输入，预测分配给三个分支的注意力权重（输出经过 Softmax 归一化）。通过空间坐标条件加权求和，自适应地选择最适合当前地点的空间尺度的特征。

### 2.2 使用 LoRA 微调图像编码器
*   **背景**：传统的 CLIP ViT-L/14 主干网络在通用图像上进行了预训练，缺乏对领域特定的细粒度街景特征（如不同地区的建筑风格、特定路标纹理、地域性植被分布等）的领域敏感度。
*   **实现**：采用低秩自适应（Low-Rank Adaptation, LoRA）技术对 ImageEncoder 进行微调。在预训练 ViT 中语义较深的最后 6 层（第 18–23 层）自注意力机制模块注入低秩分解矩阵。在避免破坏预训练视觉通用概念的同时引入街景鉴别知识。

### 2.3 基于地理距离约束的负样本采样
*   **背景**：在 InfoNCE 的对比学习计算中，除了配对真值（Positive）外，批处理中其他所有地点样本均被等价视作主负样本（Negative）。在地理定位任务特征学习时缺乏对地理位置差异距离先验（如：将相似度识别错判为相距 200km 城市所带来的惩罚，不应具有与错放至距离 9000km 区域一样的强度）。
*   **实现**：在交叉熵损失计算 Softmax logits 前引入距离阈值截断模式，剔除实际地理空间过于接近的假负例。

## 3. 项目仓库
*   https://github.com/Vitalrubbish/GeoTX

## 4. 参考文献
*   Vivanco, Vicente, Gaurav Kumar Nayak, and Mubarak Shah. "GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization." _Advances in Neural Information Processing Systems_ (2023). [arXiv:2309.16020](https://arxiv.org/abs/2309.16020)