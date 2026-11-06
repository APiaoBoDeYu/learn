# 03 · 逐行读懂 IR 反解析代码的语法

拿真实文件 `jit2engine/jit_code/backbone/000_predict_backbone.py` 全文做例子
（只有 25 行，是最短的一个入口，非常适合当作"识字课本"）：

```python
def predict_backbone(self,
    rgbs: Tensor) -> Dict[str, Tensor]:
  _0 = __torch__.covariant.common.learning_utils.torch_jit_utils.jit_compatible_tensor
  module = self.module
  _backbone_fpn_handler = module._backbone_fpn_handler
  _lateral_fpn_cores = _backbone_fpn_handler._lateral_fpn_cores
  _00 = getattr(_lateral_fpn_cores, "0")
  _conv = _00._conv
  weight = _conv.weight
  module0 = self.module
  _backbone_fpn_handler0 = module0._backbone_fpn_handler
  module1 = self.module
  _backbone_regnet_handler = module1._backbone_regnet_handler
  inputs = torch.unsqueeze(rgbs, 0)
  _1 = (_backbone_regnet_handler).forward(inputs, )
  _2, _3, _4, _5, _6, = _1
  _7 = (_backbone_fpn_handler0).forward(_2, _3, _4, _5, _6, )
  _8, _9, _10, _11, _12, = _7
  x = torch.select(_8, 0, 0)
  x0 = torch.select(_9, 0, 0)
  x1 = torch.select(_10, 0, 0)
  x2 = torch.select(_11, 0, 0)
  x3 = torch.select(_12, 0, 0)
  _13 = {"__len__": _0(5., weight, 4, ), "0": x, "1": x0, "2": x1, "3": x2, "4": x3}
  return _13
```

一行一行拆解每个"看起来奇怪"的写法。

## 3.1 `_0`, `_1`, `_2`... 这些是什么

**TorchScript 把 IR 打印成文本时给每个中间结果自动生成的临时变量名**，等价于普通编译器输出的
`%0`, `%1`, `%2`（LLVM IR 风格）。它们**没有任何语义**，纯粹是"第 N 个被计算出来的值"。
`module`、`module0`、`module1` 也是一样——你会发现代码里反复出现
`module = self.module` / `module0 = self.module` / `module1 = self.module`，
**这三行做的是完全相同的事**（取同一个属性），只是 TorchScript 的 IR 表示里，
每次"访问属性"在图里是一个独立节点，打印器忠实地给每次访问都分配了新变量名，
而不是识别出"这其实是同一个值，可以复用"。**读的时候可以完全无视这种重复**，
把 `module`/`module0`/`module1` 都在脑子里替换成 `self.module` 即可。

这是关键的心智转变：**IR 反解析代码不是给人看着舒服的代码，是 IR 图的文本序列化**。
它保真、但不"优雅"，读的时候要主动做变量名折叠。

## 3.2 `getattr(_lateral_fpn_cores, "0")` 是什么

对应源码里的 `nn.ModuleList` 或 `nn.ModuleDict`（数字字符串做 key）。
Python 里你会写 `self._lateral_fpn_cores[0]`

## 3.3 前 8 行看起来在做事，其实"什么都没做"

```python
_0 = __torch__.covariant.common.learning_utils.torch_jit_utils.jit_compatible_tensor
module = self.module
_backbone_fpn_handler = module._backbone_fpn_handler
_lateral_fpn_cores = _backbone_fpn_handler._lateral_fpn_cores
_00 = getattr(_lateral_fpn_cores, "0")
_conv = _00._conv
weight = _conv.weight
```

这一段**只是在取一个 `weight` 张量的引用**，为了给最后一行的
`_0(5., weight, 4, )` 提供一个"随便找一个已知 dtype/device 的张量"作为参数
（`jit_compatible_tensor` 这类工具函数常见用法：需要一个 reference tensor
来创建新 tensor 时保证 dtype/device 一致。
IR 反解析代码不会区分"主干逻辑"和"边缘工具调用"，你得自己识别出
`weight` 只在最后一行的 `_0(...)` 里用到、和 RegNet/FPN 的主计算无关，
才能把注意力放到真正的骨架上。

## 3.4 真正的骨架，折叠掉噪音后

把上面的技巧应用一遍，这 25 行的真实含义是：

```python
def predict_backbone(self, rgbs):
    inputs = rgbs.unsqueeze(0)                                  # 加 batch 维
    blobs = self.module._backbone_regnet_handler.forward(inputs)   # RegNet: 1 输入 → 5 blob
    feats = self.module._backbone_fpn_handler.forward(*blobs)      # FPN: 5 blob → 5 feat
    feats = [f.select(0, 0) for f in feats]                       # 去掉 batch 维（batch=1）
    return {"0": feats[0], "1": feats[1], "2": feats[2], "3": feats[3], "4": feats[4],
            "__len__": <一个记录长度=5的辅助值，写法是历史遗留的 jit_compatible_tensor 技巧>}
```

这正是 `jit2model_guide.md` 第一节里给出的结论（"从这里可以确认：RegNet 输出 5 个
blob，FPN 消费这 5 个 blob，输出也是 5 个"）——现在你知道**为什么**能得出这个结论，
以及怎么亲手从 25 行噪音里榨出这 6 行骨架。

## 3.5 更复杂的例子：depth 的调用链是怎么"看起来乱"的

打开 `jit2engine/jit_code/depth/000_predict_depth.py` 前 90 行，会看到几十行
类似这样的堆叠：

```python
module = self.module
_backbone_fpn_handler = module._backbone_fpn_handler
_lateral_fpn_cores = _backbone_fpn_handler._lateral_fpn_cores
_00 = getattr(_lateral_fpn_cores, "0")
_conv = _00._conv
weight = _conv.weight
module0 = self.module
_normals = module0._normals
module1 = self.module
_edgefpn = module1._edgefpn
module2 = self.module
_all_depthpass = module2._all_depthpass
_50 = getattr(_all_depthpass, "5")
_depth_valid_pred_head = _50._depth_valid_pred_head
...
```

这不是"depth 模型比 backbone 复杂 20 倍"，而是**同一种噪音模式重复了 20 次**：
每一行 `moduleN = self.module` 都是同一个属性访问的重复展开，
每一个 `getattr(_all_depthpass, "5")` 都只是 `self.module._all_depthpass[5]`。
**读大模型的 JIT 代码时，第一件事永远是：用脑子（或写个小脚本）把这些
`moduleN = self.module` / `getattr(x, "N")` 折叠掉，只保留"谁调用了谁的 forward"这条主线**，
否则会被这些结构性噪音淹没，误以为骨架真的有几十个独立步骤。

`jit_learn/labs/03_module_hierarchy_and_mangling.py` 里有一个小工具函数，
教你怎么用正则/AST 自动把这类噪音行折叠，生成一份"骨架视图"。

## 3.6 dict 返回值的 `"__len__"` key 是什么

`_13 = {"__len__": _0(5., weight, 4, ), "0": x, ...}` 这种写法在 covariant 项目里
反复出现，本质是给 TorchScript 的 `Dict[str, Tensor]` 返回值类型**塞进一个"隧道编码"
的长度信息**，因为 TorchScript 的 `Dict` 不能直接塞一个 Python int 作为 value
（value 类型必须和其他 value 一致，这里被声明为 `Tensor`），于是用
`jit_compatible_tensor(5., weight, 4)` 构造出一个"看起来像 Tensor、实际编码了整数 5"
的值，塞进 `"__len__"` 这个特殊 key，供调用方（比如 ONNX 导出脚本或 C++ 推理端）
读取这个 dict 到底有几个真正的 tensor 输出。**这是 covariant 项目自己的约定，
不是 TorchScript 语言特性**——遇到不认识的辅助 key/辅助函数，
先假设它是"项目自己发明的、绕过 TorchScript 类型限制的技巧"，
去 `covariant/common/learning_utils/torch_jit_utils.py` 查函数定义，
而不是当成 TorchScript 标准语法死磕。

## 3.7 小结：读 IR 反解析代码的三条固定动作

1. **折叠 `moduleN = self.module`、`_NN = getattr(x, "K")` 这类重复访问**，
   还原成一条 `self.module._a._b[c]` 式的属性链。
2. **区分"主干张量流"和"边缘工具调用"**——凡是只被用来构造某个特殊返回值/元信息
   （像 `weight` 只喂给 `jit_compatible_tensor`）的变量，先标记为噪音，不要当成主逻辑。
3. **遇到不认识的函数/写法，先查 `__torch__.xxx.yyy` 对应的真实模块路径**
   （IR 反解析代码里函数全名都是完整路径，比如
   `__torch__.covariant.common.learning_utils.torch_jit_utils.jit_compatible_tensor`），
   这条路径直接对应源码里 `covariant/common/learning_utils/torch_jit_utils.py`，
   永远可以按路径去源码里找到真正的实现和文档。

下一章讲"为什么即使读懂了每一行语法，还是拿不到超参数"——这是语义层面的限制，
不是语法层面的。
