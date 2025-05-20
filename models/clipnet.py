import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import open_clip
dict_pretrain = {
    'clipL14openai'     : ('ViT-L-14', 'openai'),
    'clipL14laion400m'  : ('ViT-L-14', 'laion400m_e32'),
    'clipL14laion2B'    : ('ViT-L-14', 'laion2b_s32b_b82k'),
    'clipL14datacomp'   : ('ViT-L-14', 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'),
    'clipL14commonpool' : ('ViT-L-14', "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K", 'open_clip_pytorch_model.bin'),
    'clipaL14datacomp'  : ('ViT-L-14-CLIPA', 'datacomp1b'),
    'cocaL14laion2B'    : ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
    'clipg14laion2B'    : ('ViT-g-14', 'laion2b_s34b_b88k'),
    'eva2L14merged2b'   : ('EVA02-L-14', 'merged2b_s4b_b131k'),
    'clipB16laion2B'    : ('ViT-B-16', 'laion2b_s34b_b88k'),
}
# 预训练模型权重的路径
pretrained_weights_path = '../weights/open_clip_pytorch_model.bin'  # 替换为您的权重文件路径
# 加载预训练的 CLIP 模型
try:
    model_name = 'ViT-L-14'
    backbone = open_clip.create_model(model_name, pretrained=pretrained_weights_path)
    print(f"Loaded CLIP model: {model_name}")
except Exception as e:
    print(f"Error loading CLIP model: {e}")

# class SelfAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(in_dim, in_dim)
#         self.key = nn.Linear(in_dim, in_dim)
#         self.value = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         batch_size, num_features, in_dim = x.size()
#
#         proj_query = self.query(x)  # [batch_size, num_features, in_dim]
#         proj_key = self.key(x)  # [batch_size, num_features, in_dim]
#         proj_value = self.value(x)  # [batch_size, num_features, in_dim]
#
#         # 计算能量
#         energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
#         attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]
#
#         # 应用注意力机制
#         out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]
#
#         # 将输出展平为 [batch_size, 1, in_dim]
#         out = out.mean(dim=1, keepdim=True)  # 取特征图间的平均值
#
#         return out
# # OpenClipLinear 模型定义
# class OpenClipLinear(nn.Module):
#     def __init__(self, normalize=True, next_to_last=False, pretrained_model_path=None):
#         super(OpenClipLinear, self).__init__()
#
#         self.backbone = open_clip.create_model('ViT-L-14', pretrained=pretrained_model_path)
#
#         if next_to_last:
#             self.num_features = self.backbone.visual.proj.shape[0]  # 倒数第二层的特征数量
#             self.backbone.visual.proj = None  # 移除最后的投影层
#         else:
#             self.num_features = self.backbone.visual.output_dim
#
#         self.normalize = normalize
#         self.self_attention = SelfAttention(self.num_features)
#         self.fc = nn.Linear(self.num_features, 1)  # 输出一个概率值
#
#     def to(self, *args, **kwargs):
#         self.backbone.to(*args, **kwargs)
#         super(OpenClipLinear, self).to(*args, **kwargs)
#         return self
#
#     def forward_features(self, x):
#         with torch.no_grad():
#             self.backbone.eval()
#             features = self.backbone.encode_image(x, normalize=self.normalize)
#         return features
#
#     def forward(self, x):
#         # print(f"Input to OpenClipLinear: {x.shape}")  # 打印输入尺寸
#
#         features_list = [self.forward_features(img) for img in x]
#
#         # 确保 features_list 中的每个特征都是 (batch_size, 1024) 形状
#         # for i, feature in enumerate(features_list):
#         #     print(f"Feature {i} shape: {feature.shape}")
#
#         # 特征融合，这里使用 stack
#         # 确保 stack 的维度正确，应该是 (batch_size, 3, 1024)
#         features = torch.stack(features_list, dim=0)  # 沿着新的维度堆叠
#
#         # print(f"Stacked features shape: {features.shape}")  # 打印堆叠后的特征尺寸
#
#         # 直接将三维特征图输入到自注意力层
#         features = self.self_attention(features)
#         features = features.view(features.size(0), -1)
#         output = self.fc(features)
#         # print(f"Output from OpenClipLinear: {output.shape}")  # 打印输出尺寸
#         return output
#
#     def freeze_clip(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#
#         for param in self.self_attention.parameters():
#             param.requires_grad = True
#         for param in self.fc.parameters():
#             param.requires_grad = True
#
#     def unfreeze_clip(self):
#         for param in self.parameters():
#             param.requires_grad = True




# import torch
# import torch.nn as nn
#
# # 自注意力模块
# class SelfAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(in_dim, in_dim)
#         self.key = nn.Linear(in_dim, in_dim)
#         self.value = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         batch_size, num_features, in_dim = x.size()
#
#         proj_query = self.query(x)  # [batch_size, num_features, in_dim]
#         proj_key = self.key(x)  # [batch_size, num_features, in_dim]
#         proj_value = self.value(x)  # [batch_size, num_features, in_dim]
#
#         # 计算能量
#         energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
#         attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]
#
#         # 应用注意力机制
#         out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]
#
#         # 将输出与输入相加并乘以gamma
#         out = self.gamma * out + x
#
#         return out
#
# # 注意力特征融合模块（AFF）
# class AFF(nn.Module):
#     def __init__(self, in_dim):
#         super(AFF, self).__init__()
#         self.query = nn.Linear(in_dim, in_dim)
#         self.key = nn.Linear(in_dim, in_dim)
#         self.value = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x1, x2):
#         batch_size, num_features, in_dim = x1.size()
#
#         proj_query = self.query(x1)  # [batch_size, num_features, in_dim]
#         proj_key = self.key(x2)  # [batch_size, num_features, in_dim]
#         proj_value = self.value(x2)  # [batch_size, num_features, in_dim]
#
#         # 计算能量
#         energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
#         attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]
#
#         # 应用注意力机制
#         out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]
#
#         # 将输出与输入相加并乘以gamma
#         out = self.gamma * out + x1
#
#         return out
#
# # OpenClipLinear 模型定义
# class OpenClipLinear(nn.Module):
#     def __init__(self, normalize=True, next_to_last=False, pretrained_model_path=None):
#         super(OpenClipLinear, self).__init__()
#
#         self.backbone = open_clip.create_model('ViT-L-14', pretrained=pretrained_model_path)
#
#         if next_to_last:
#             self.num_features = self.backbone.visual.proj.shape[0]  # 倒数第二层的特征数量
#             self.backbone.visual.proj = None  # 移除最后的投影层
#         else:
#             self.num_features = self.backbone.visual.output_dim
#
#         self.normalize = normalize
#         self.self_attention = SelfAttention(self.num_features)
#         self.aff = AFF(self.num_features)
#         self.fc = nn.Linear(self.num_features, 1)  # 输出一个概率值
#
#     def to(self, *args, **kwargs):
#         self.backbone.to(*args, **kwargs)
#         super(OpenClipLinear, self).to(*args, **kwargs)
#         return self
#
#     def forward_features(self, x):
#         with torch.no_grad():
#             self.backbone.eval()
#             features = self.backbone.encode_image(x, normalize=self.normalize)
#         return features
#
#     def forward(self, x):
#         # x 是一个包含 batch_size 个特征的列表，每个特征的形状是 [3, 1024]
#         batch_size = len(x)
#         features_list = [self.forward_features(img) for img in x]
#
#         # # 打印每个特征的形状
#         # for i, feature in enumerate(features_list):
#         #     print(f"Feature {i} shape: {feature.shape}")
#
#         # 提取局部特征和全局特征
#         fused_features = []
#         for i in range(batch_size):
#             # 提取三个特征
#             local_feature1 = features_list[i][0, :]  # 形状为 [1024]
#             local_feature2 = features_list[i][1, :]  # 形状为 [1024]
#             global_feature = features_list[i][2, :]  # 形状为 [1024]
#
#             # 调整形状以适配 AFF 模块
#             local_feature1 = local_feature1.unsqueeze(0).unsqueeze(0)  # 形状为 [1, 1, 1024]
#             local_feature2 = local_feature2.unsqueeze(0).unsqueeze(0)  # 形状为 [1, 1, 1024]
#             global_feature = global_feature.unsqueeze(0).unsqueeze(0)  # 形状为 [1, 1, 1024]
#
#             # 局部特征融合
#             fused_local = self.aff(local_feature1, local_feature2)
#
#             # 全局特征融合
#             fused_feature = self.aff(fused_local, global_feature).squeeze(1)
#             # print(fused_feature.shape)
#             fused_features.append(fused_feature)
#
#         # 将融合后的特征堆叠成一个张量
#         fused_features = torch.stack(fused_features, dim=0)  # 形状为 [batch_size, 1024]
#
#         # 自注意力机制
#
#         fused_features = self.self_attention(fused_features)
#         fused_features = fused_features.squeeze(1)  # 移除多余的维度
#
#         # 打印最终融合特征的形状
#         # print(f"Fused features shape: {fused_features.shape}")
#
#         # 全连接层
#         output = self.fc(fused_features)
#         return output
#
#     def freeze_clip(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#
#         for param in self.self_attention.parameters():
#             param.requires_grad = True
#         for param in self.fc.parameters():
#             param.requires_grad = True
#
#     def unfreeze_clip(self):
#         for param in self.parameters():
#             param.requires_grad = True
# class AFF(nn.Module):
#     def __init__(self, in_dim):
#         super(AFF, self).__init__()
#         self.query = nn.Linear(in_dim, in_dim)
#         self.key = nn.Linear(in_dim, in_dim)
#         self.value = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x1, x2):
#         batch_size, num_features, in_dim = x1.size()
#
#         proj_query = self.query(x1)  # [batch_size, num_features, in_dim]
#         proj_key = self.key(x2)  # [batch_size, num_features, in_dim]
#         proj_value = self.value(x2)  # [batch_size, num_features, in_dim]
#
#         # 计算能量
#         energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
#         attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]
#
#         # 应用注意力机制
#         out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]
#
#         # 将输出与输入相加并乘以gamma
#         out = self.gamma * out + x1
#
#         return out
#
# class OpenClipLinear(nn.Module):
#     def __init__(self, normalize=True, next_to_last=False, pretrained_model_path=None, num_classes=2):
#         super(OpenClipLinear, self).__init__()
#
#         # CLIP 特征提取器
#         self.clip_model = open_clip.create_model('ViT-L-14', pretrained=pretrained_model_path)
#         if next_to_last:
#             self.num_features = self.clip_model.visual.proj.shape[0]
#             self.clip_model.visual.proj = None
#         else:
#             self.num_features = self.clip_model.visual.output_dim
#         self.normalize = normalize
#
#         # 注意力特征融合模块
#         self.aff = AFF(self.num_features)
#
#         # 定义全连接分类器
#         self.classifier = nn.Linear(self.num_features, 1)
#
#     def forward_features(self, x):
#         with torch.no_grad():
#             self.clip_model.eval()
#             features = self.clip_model.encode_image(x, normalize=self.normalize)
#         return features
#
#     def forward(self, x):
#         batch_size = len(x)
#         features_list = [self.forward_features(img) for img in x]
#
#         fused_features = []
#         for i in range(batch_size):
#             local_feature1 = features_list[i][0, :].unsqueeze(0).unsqueeze(0)
#             local_feature2 = features_list[i][1, :].unsqueeze(0).unsqueeze(0)
#             global_feature = features_list[i][2, :].unsqueeze(0).unsqueeze(0)
#
#             fused_local = self.aff(local_feature1, local_feature2)
#             fused_feature = self.aff(fused_local, global_feature).squeeze(1)
#             fused_features.append(fused_feature)
#
#         fused_features = torch.stack(fused_features, dim=0)  # 形状为 [batch_size, 1024]
#
#         # 直接使用全连接层进行分类
#         output = self.classifier(fused_features)  # [batch_size, num_classes]
#
#         return output
#
#     def freeze_clip(self):
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
#
#         for param in self.aff.parameters():
#             param.requires_grad = True
#         for param in self.classifier.parameters():
#             param.requires_grad = True
#
#     def unfreeze_clip(self):
#         for param in self.parameters():
#             param.requires_grad = True
# import torch
# import open_clip
#
#
# # 定义测试函数
# def test_model(pretrained_model_path):
#     # 初始化模型，设置 next_to_last=True 以使用倒数第二层的输出
#     model = OpenClipLinear(next_to_last=True, pretrained_model_path=pretrained_model_path)
#
#     # 将模型设置为评估模式
#     model.eval()
#
#     # 创建一个假的输入张量，模拟三个224x224的图片
#     # 假设输入图片的batch size为1，通道数为3（RGB图像）
#     # 这里我们需要确保批次大小与模型期望的批次大小相匹配
#     dummy_input = torch.randn(1, 3, 224, 224)  # 确保维度正确
#
#     # 通过模型传递dummy_input并获取输出
#     with torch.no_grad():
#         output = model([dummy_input])  # 注意这里传递的是一个列表，因为模型期望多个图像输入
#
#     # 打印输出的维度
#     print("Output shape:", output.shape)
#
#     # 打印特征的维度
#     features_list = [model.forward_features(img) for img in [dummy_input]]  # 同样传递一个列表
#     print("Features shape:", features_list[0].shape)
#
#
# # 调用测试函数
# pretrained_weights_path = '../weights/open_clip_pytorch_model.bin'  # 替换为您的权重文件路径
# test_model(pretrained_weights_path)
# def test_model(pretrained_model_path, batch_size):
#     # 初始化模型，设置 next_to_last=True 以使用倒数第二层的输出
#     model = OpenClipLinear(next_to_last=True, pretrained_model_path=pretrained_model_path)
#
#     # 将模型设置为评估模式
#     model.eval()
#
#     # 创建一个假的输入张量，模拟指定批量大小的224x224的图片
#     # 假设输入图片的通道数为3（RGB图像）
#     dummy_input = torch.randn(batch_size, 3, 224, 224)  # 确保维度正确
#
#     # 通过模型传递dummy_input并获取输出
#     with torch.no_grad():
#         # 将单个张量包装成列表，因为模型可能期望多个图像输入
#         output = model([dummy_input])
#
#     # 打印输出的维度
#     print("Output shape:", output.shape)
#
#     # 打印特征的维度
#     features = model.forward_features(dummy_input)
#     print("Features shape:", features.shape)
#
# # 调用测试函数
# pretrained_weights_path = '../weights/open_clip_pytorch_model.bin'  # 替换为您的权重文件路径
# batch_size = 8  # 设置您想要的批量大小
# test_model(pretrained_weights_path, batch_size)






#3 # 迭代注意力特征融合模块（iAFF）
# class iAFF(nn.Module):
#     def __init__(self, in_dim, num_iterations=2):
#         super(iAFF, self).__init__()
#         self.num_iterations = num_iterations
#         self.aff_layers = nn.ModuleList([AFF(in_dim) for _ in range(num_iterations)])
#
#     def forward(self, x1, x2):
#         # 初始特征融合
#         fused_feature = x1 + x2
#
#         # 迭代应用注意力特征融合
#         for i in range(self.num_iterations):
#             fused_feature = self.aff_layers[i](x1, fused_feature)
#
#         return fused_feature
#
# # 注意力特征融合模块（AFF）
# class AFF(nn.Module):
#     def __init__(self, in_dim):
#         super(AFF, self).__init__()
#         self.query = nn.Linear(in_dim, in_dim)
#         self.key = nn.Linear(in_dim, in_dim)
#         self.value = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x1, x2):
#         batch_size, num_features, in_dim = x1.size()
#
#         proj_query = self.query(x1)  # [batch_size, num_features, in_dim]
#         proj_key = self.key(x2)  # [batch_size, num_features, in_dim]
#         proj_value = self.value(x2)  # [batch_size, num_features, in_dim]
#
#         # 计算能量
#         energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
#         attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]
#
#         # 应用注意力机制
#         out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]
#
#         # 将输出与输入相加并乘以gamma
#         out = self.gamma * out + x1
#
#         return out
#
# class OpenClipLinear(nn.Module):
#     def __init__(self, normalize=True, next_to_last=False, pretrained_model_path=None, num_classes=1):
#         super(OpenClipLinear, self).__init__()
#
#         # CLIP 特征提取器
#         self.clip_model = open_clip.create_model('ViT-L-14', pretrained=pretrained_model_path)
#         if next_to_last:
#             self.num_features = self.clip_model.visual.proj.shape[0]
#             self.clip_model.visual.proj = None
#         else:
#             self.num_features = self.clip_model.visual.output_dim
#         self.normalize = normalize
#
#         # 迭代注意力特征融合模块
#         self.iaff = iAFF(self.num_features)
#
#         # 定义全连接分类器
#         self.classifier = nn.Linear(self.num_features, 1)
#
#     def forward_features(self, x):
#         with torch.no_grad():
#             self.clip_model.eval()
#             features = self.clip_model.encode_image(x, normalize=self.normalize)
#         return features
#
#     def forward(self, x):
#         batch_size = len(x)
#         features_list = [self.forward_features(img) for img in x]
#
#         fused_features = []
#         for i in range(batch_size):
#             local_feature1 = features_list[i][0, :].unsqueeze(0).unsqueeze(0)
#             local_feature2 = features_list[i][1, :].unsqueeze(0).unsqueeze(0)
#             global_feature = features_list[i][2, :].unsqueeze(0).unsqueeze(0)
#
#             # 使用 iAFF 进行特征融合
#             fused_feature = self.iaff(local_feature1, local_feature2)
#             fused_feature = self.iaff(fused_feature, global_feature).squeeze(1)
#             fused_features.append(fused_feature)
#
#         fused_features = torch.stack(fused_features, dim=0)  # 形状为 [batch_size, 1024]
#
#         # 直接使用全连接层进行分类
#         output = self.classifier(fused_features)  # [batch_size, num_classes]
#
#         return output
#
#     def freeze_clip(self):
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
#
#         for param in self.iaff.parameters():
#             param.requires_grad = True
#         for param in self.classifier.parameters():
#             param.requires_grad = True
#
#     def unfreeze_clip(self):
#         for param in self.parameters():
#             param.requires_grad = True

#4
# import torch
# import torch.nn as nn
# import open_clip
#
# class AFF(nn.Module):
#     def __init__(self, in_dim):
#         super(AFF, self).__init__()
#         self.query = nn.Linear(in_dim, in_dim)
#         self.key = nn.Linear(in_dim, in_dim)
#         self.value = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x1, x2):
#         batch_size, num_features, in_dim = x1.size()
#
#         proj_query = self.query(x1)  # [batch_size, num_features, in_dim]
#         proj_key = self.key(x2)  # [batch_size, num_features, in_dim]
#         proj_value = self.value(x2)  # [batch_size, num_features, in_dim]
#
#         # 计算能量
#         energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
#         attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]
#
#         # 应用注意力机制
#         out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]
#
#         # 将输出与输入相加并乘以gamma
#         out = self.gamma * out + x1
#
#         return out
#
# class TransformerClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes=1, num_heads=8, num_layers=2, dim_feedforward=2048):
#         super(TransformerClassifier, self).__init__()
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(
#                 d_model=input_dim,
#                 nhead=num_heads,
#                 dim_feedforward=dim_feedforward,
#                 batch_first=True
#             ),
#             num_layers=num_layers
#         )
#         self.classifier = nn.Linear(input_dim, num_classes)
#
#     def forward(self, x):
#         # 添加一个可学习的分类token
#         cls_token = nn.Parameter(torch.randn(1, 1, x.size(-1))).to(x.device)
#         cls_token = cls_token.expand(x.size(0), -1, -1)
#         x = torch.cat([cls_token, x], dim=1)
#
#         # 通过Transformer编码器
#         x = self.transformer(x)
#
#         # 使用cls_token的输出进行分类
#         cls_output = x[:, 0, :]
#         output = self.classifier(cls_output)
#
#         return output
#
# class OpenClipLinear(nn.Module):
#     def __init__(self, normalize=True, next_to_last=False, pretrained_model_path=None, num_classes=1):
#         super(OpenClipLinear, self).__init__()
#
#         # CLIP 特征提取器
#         self.clip_model = open_clip.create_model('ViT-L-14', pretrained=pretrained_model_path)
#         if next_to_last:
#             self.num_features = self.clip_model.visual.proj.shape[0]
#             self.clip_model.visual.proj = None
#         else:
#             self.num_features = self.clip_model.visual.output_dim
#         self.normalize = normalize
#
#         # 注意力特征融合模块
#         self.aff = AFF(self.num_features)
#
#         # 定义Transformer分类器
#         self.transformer_classifier = TransformerClassifier(
#             input_dim=self.num_features,
#             num_classes=num_classes,
#             num_heads=8,
#             num_layers=2,
#             dim_feedforward=2048
#         )
#
#     def forward_features(self, x):
#         with torch.no_grad():
#             self.clip_model.eval()
#             features = self.clip_model.encode_image(x, normalize=self.normalize)
#         return features
#
#     def forward(self, x):
#         batch_size = len(x)
#         features_list = [self.forward_features(img) for img in x]
#
#         fused_features = []
#         for i in range(batch_size):
#             local_feature1 = features_list[i][0, :].unsqueeze(0).unsqueeze(0)
#             local_feature2 = features_list[i][1, :].unsqueeze(0).unsqueeze(0)
#             global_feature = features_list[i][2, :].unsqueeze(0).unsqueeze(0)
#             # print(global_feature.shape)
#             # print(local_feature1.shape)
#             # print(local_feature2.shape)
#             fused_local = self.aff(local_feature1, local_feature2)
#             fused_feature = self.aff(fused_local, global_feature).squeeze(1)
#             fused_features.append(fused_feature)
#             # print(f"fused_feature {i} shape: {fused_feature.shape}")
#
#         fused_features = torch.stack(fused_features, dim=0)  # 形状为 [batch_size, 1024]
#
#         # 使用Transformer分类器进行分类
#         # print(fused_features.shape)
#         output = self.transformer_classifier(fused_features)  # [batch_size, 1]
#         # print(output.shape)
#         return output
#
#     def freeze_clip(self):
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
#
#         for param in self.aff.parameters():
#             param.requires_grad = True
#         for param in self.transformer_classifier.parameters():
#             param.requires_grad = True
#
#     def unfreeze_clip(self):
#         for param in self.parameters():
#             param.requires_grad = True


# import torch
# import torch.nn as nn
# import open_clip
#
# class MultiScaleAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(MultiScaleAttention, self).__init__()
#         self.scale1 = nn.Linear(in_dim, in_dim)
#         self.scale2 = nn.Linear(in_dim, in_dim)
#         self.scale3 = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         # 多尺度特征提取
#         s1 = self.scale1(x)
#         s2 = self.scale2(x)
#         s3 = self.scale3(x)
#
#         # 计算多尺度注意力
#         attention = torch.sigmoid(s1 + s2 + s3)
#
#         return attention
#
# class AFF(nn.Module):
#     def __init__(self, in_dim):
#         super(AFF, self).__init__()
#         self.query = nn.Linear(in_dim, in_dim)
#         self.key = nn.Linear(in_dim, in_dim)
#         self.value = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.bn = nn.BatchNorm1d(in_dim)
#         self.ms_attention = MultiScaleAttention(in_dim)
#
#     def forward(self, x1, x2):
#         batch_size, num_features, in_dim = x1.size()
#
#         proj_query = self.query(x1)  # [batch_size, num_features, in_dim]
#         proj_key = self.key(x2)  # [batch_size, num_features, in_dim]
#         proj_value = self.value(x2)  # [batch_size, num_features, in_dim]
#
#         # 计算能量
#         energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
#         attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]
#
#         # 应用注意力机制
#         out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]
#
#         # 多尺度通道注意力
#         ms_attention = self.ms_attention(out)
#         out = out * ms_attention
#
#
#         # 将输出与输入相加并乘以 gamma
#         out = self.gamma * out + x1
#
#         return out
#
# class OpenClipLinear(nn.Module):
#     def __init__(self, normalize=True, next_to_last=False, pretrained_model_path=None, num_classes=2):
#         super(OpenClipLinear, self).__init__()
#
#         # CLIP 特征提取器
#         self.clip_model = open_clip.create_model('ViT-L-14', pretrained=pretrained_model_path)
#         if next_to_last:
#             self.num_features = self.clip_model.visual.proj.shape[0]
#             self.clip_model.visual.proj = None
#         else:
#             self.num_features = self.clip_model.visual.output_dim
#         self.normalize = normalize
#
#         # 注意力特征融合模块
#         self.aff = AFF(self.num_features)
#
#         # 定义全连接分类器
#         self.classifier = nn.Linear(self.num_features, 1)
#
#     def forward_features(self, x):
#         with torch.no_grad():
#             self.clip_model.eval()
#             features = self.clip_model.encode_image(x, normalize=self.normalize)
#         return features
#
#     def forward(self, x):
#         batch_size = len(x)
#         features_list = [self.forward_features(img) for img in x]
#
#         fused_features = []
#         for i in range(batch_size):
#             local_feature1 = features_list[i][0, :].unsqueeze(0).unsqueeze(0)
#             local_feature2 = features_list[i][1, :].unsqueeze(0).unsqueeze(0)
#             global_feature = features_list[i][2, :].unsqueeze(0).unsqueeze(0)
#
#             fused_local = self.aff(local_feature1, local_feature2)
#             fused_feature = self.aff(fused_local, global_feature).squeeze(1)
#             fused_features.append(fused_feature)
#
#         fused_features = torch.stack(fused_features, dim=0)  # 形状为 [batch_size, 1024]
#
#         # 直接使用全连接层进行分类
#         output = self.classifier(fused_features)  # [batch_size, num_classes]
#
#         return output
#
#     def freeze_clip(self):
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
#
#         for param in self.aff.parameters():
#             param.requires_grad = True
#         for param in self.classifier.parameters():
#             param.requires_grad = True
#
#     def unfreeze_clip(self):
#         for param in self.parameters():
#             param.requires_grad = True
import torch
import torch.nn as nn
import open_clip

class MultiScaleAttention(nn.Module):
    def __init__(self, in_dim):
        super(MultiScaleAttention, self).__init__()
        self.scale1 = nn.Linear(in_dim, in_dim)
        self.scale2 = nn.Linear(in_dim, in_dim)
        self.scale3 = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 多尺度特征提取
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)

        # 计算多尺度注意力
        attention = torch.sigmoid(s1 + s2 + s3)

        return attention

class AFF(nn.Module):
    def __init__(self, in_dim):
        super(AFF, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm1d(in_dim)
        self.ms_attention = MultiScaleAttention(in_dim)

    def forward(self, x1, x2):
        batch_size, num_features, in_dim = x1.size()

        proj_query = self.query(x1)  # [batch_size, num_features, in_dim]
        proj_key = self.key(x2)  # [batch_size, num_features, in_dim]
        proj_value = self.value(x2)  # [batch_size, num_features, in_dim]

        # 计算能量
        energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
        attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]

        # 应用注意力机制
        out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]

        # 多尺度通道注意力
        ms_attention = self.ms_attention(out)
        out = out * ms_attention


        # 将输出与输入相加并乘以 gamma
        out = self.gamma * out + x1

        return out

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, num_features, in_dim = x.size()

        proj_query = self.query(x)  # [batch_size, num_features, in_dim]
        proj_key = self.key(x)  # [batch_size, num_features, in_dim]
        proj_value = self.value(x)  # [batch_size, num_features, in_dim]

        # 计算能量
        energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
        attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]

        # 应用注意力机制
        out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]

        # 将输出与输入相加并乘以gamma
        out = self.gamma * out + x

        return out


class OpenClipLinear(nn.Module):
    def __init__(self, normalize=True, next_to_last=False, pretrained_model_path=None, num_classes=2):
        super(OpenClipLinear, self).__init__()

        # CLIP 特征提取器
        self.clip_model = open_clip.create_model('ViT-L-14', pretrained=pretrained_model_path)
        if next_to_last:
            self.num_features = self.clip_model.visual.proj.shape[0]
            self.clip_model.visual.proj = None
        else:
            self.num_features = self.clip_model.visual.output_dim
        self.normalize = normalize

        # 注意力特征融合模块
        self.aff = AFF(self.num_features)

        # SelfAttention 模块
        self.self_attention = SelfAttention(self.num_features)

        # 定义全连接分类器
        self.classifier = nn.Linear(self.num_features, 1)

    def forward_features(self, x):
        with torch.no_grad():
            self.clip_model.eval()
            features = self.clip_model.encode_image(x, normalize=self.normalize)
        return features

    def forward(self, x):
        batch_size = len(x)
        features_list = [self.forward_features(img) for img in x]

        fused_features = []
        for i in range(batch_size):
            local_feature1 = features_list[i][0, :].unsqueeze(0).unsqueeze(0)
            local_feature2 = features_list[i][1, :].unsqueeze(0).unsqueeze(0)
            global_feature = features_list[i][2, :].unsqueeze(0).unsqueeze(0)

            fused_local = self.aff(local_feature1, local_feature2)
            fused_feature = self.aff(fused_local, global_feature).squeeze(1)
            fused_features.append(fused_feature)

        fused_features = torch.stack(fused_features, dim=0)  # 形状为 [batch_size, 1024]

        # 应用 SelfAttention

        fused_features = self.self_attention(fused_features)


        # 直接使用全连接层进行分类
        output = self.classifier(fused_features)  # [batch_size, num_classes]

        return output



    def freeze_clip(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

        for param in self.aff.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_clip(self):
        for param in self.parameters():
            param.requires_grad = True

# import torch
# import torch.nn as nn
# import open_clip
#
# class MultiScaleAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(MultiScaleAttention, self).__init__()
#         self.scale1 = nn.Linear(in_dim, in_dim)
#         self.scale2 = nn.Linear(in_dim, in_dim)
#         self.scale3 = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         # 多尺度特征提取
#         s1 = self.scale1(x)
#         s2 = self.scale2(x)
#         s3 = self.scale3(x)
#
#         # 计算多尺度注意力
#         attention = torch.sigmoid(s1 + s2 + s3)
#
#         return attention
#
# class AFF(nn.Module):
#     def __init__(self, in_dim):
#         super(AFF, self).__init__()
#         self.query = nn.Linear(in_dim, in_dim)
#         self.key = nn.Linear(in_dim, in_dim)
#         self.value = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.bn = nn.BatchNorm1d(in_dim)
#         self.ms_attention = MultiScaleAttention(in_dim)
#
#     def forward(self, x1, x2):
#         batch_size, num_features, in_dim = x1.size()
#
#         proj_query = self.query(x1)  # [batch_size, num_features, in_dim]
#         proj_key = self.key(x2)  # [batch_size, num_features, in_dim]
#         proj_value = self.value(x2)  # [batch_size, num_features, in_dim]
#
#         # 计算能量
#         energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)  # [batch_size, num_features, num_features]
#         attention = torch.softmax(energy, dim=-1)  # [batch_size, num_features, num_features]
#
#         # 应用注意力机制
#         out = torch.einsum('bij,bjk->bik', attention, proj_value)  # [batch_size, num_features, in_dim]
#
#         # 多尺度通道注意力
#         ms_attention = self.ms_attention(out)
#         out = out * ms_attention
#
#
#         # 将输出与输入相加并乘以 gamma
#         out = self.gamma * out + x1
#
#         return out
#
# class ResnetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResnetBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels)
#             )
#
#     def forward(self, x):
#         identity = self.downsample(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += identity
#         out = self.relu(out)
#         return out
#
# class ResNet50(nn.Module):
#     def __init__(self, num_classes=1):
#         super(ResNet50, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为1
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(64, 3)
#         self.layer2 = self._make_layer(128, 4, stride=2)
#         self.layer3 = self._make_layer(256, 6, stride=2)
#         self.layer4 = self._make_layer(512, 3, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(512, num_classes)
#
#     def _make_layer(self, out_channels, blocks, stride=1):
#         layers = []
#         layers.append(ResnetBlock(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels
#         for _ in range(1, blocks):
#             layers.append(ResnetBlock(self.in_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
#
# class OpenClipLinear(nn.Module):
#     def __init__(self, normalize=True, next_to_last=False, pretrained_model_path=None, num_classes=1):
#         super(OpenClipLinear, self).__init__()
#
#         # CLIP 特征提取器
#         self.clip_model = open_clip.create_model('ViT-L-14', pretrained=pretrained_model_path)
#         if next_to_last:
#             self.num_features = self.clip_model.visual.proj.shape[0]
#             self.clip_model.visual.proj = None
#         else:
#             self.num_features = self.clip_model.visual.output_dim
#         self.normalize = normalize
#
#         # 注意力特征融合模块
#         self.aff = AFF(self.num_features)
#
#         # 1D ResNet50
#         self.resnet50 = ResNet50(num_classes=num_classes)
#
#     def forward_features(self, x):
#         with torch.no_grad():
#             self.clip_model.eval()
#             features = self.clip_model.encode_image(x, normalize=self.normalize)
#         return features
#
#     def forward(self, x):
#         batch_size = len(x)
#         features_list = [self.forward_features(img) for img in x]
#
#         fused_features = []
#         for i in range(batch_size):
#             local_feature1 = features_list[i][0, :].unsqueeze(0).unsqueeze(0)
#             local_feature2 = features_list[i][1, :].unsqueeze(0).unsqueeze(0)
#             global_feature = features_list[i][2, :].unsqueeze(0).unsqueeze(0)
#
#             fused_local = self.aff(local_feature1, local_feature2)
#             fused_feature = self.aff(fused_local, global_feature).squeeze(1)
#             fused_features.append(fused_feature)
#
#         fused_features = torch.stack(fused_features, dim=0)  # 形状为 [batch_size, 1024]
#
#         # 使用1D ResNet50进行特征提取和分类
#         output = self.resnet50(fused_features)  # [batch_size, 1]
#
#         return output
#
#     def freeze_clip(self):
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
#
#         for param in self.aff.parameters():
#             param.requires_grad = True
#         for param in self.resnet50.parameters():
#             param.requires_grad = True
#
#     def unfreeze_clip(self):
#         for param in self.parameters():
#             param.requires_grad = True