# 05 · 动手实验室：把前四章的结论在自己手上验证一遍

这一章不是新知识，是"操作清单"——把 `jit_learn/labs/` 下四个脚本串起来讲一遍，
每个脚本对应哪一章、验证了哪句话、为什么值得亲手跑。**读文档只能理解，
跑一遍才能真正建立直觉**，尤其是"mangle 数量"、"Final 常量"这类东西，
自己在真实数据上看到具体数字，比记住一句结论有用得多。

## 5.1 环境要求

这些脚本只依赖标准库 + PyTorch，不需要 covariant 环境、不需要 Bazel、
不需要 GPU。仓库里的 Python 环境在 `/opt/venv`：

```bash
/opt/venv/bin/python3 --version   # 应该是 3.12.x
/opt/venv/bin/python3 -c "import torch; print(torch.__version__)"   # 2.9.1（写这份文档时的版本）
```

后面所有命令都用这个解释器跑。

## 5.2 四个 lab，按顺序跑一遍

```bash
cd /root/brain/brain-service/jit2engine/jit_learn/labs

# Lab 01：trace vs script 的语义区别 —— 对应第 01、04 章
/opt/venv/bin/python3 01_trace_vs_script.py

# Lab 02：解剖真实的 unified.pt zip 文件 —— 对应第 02 章
/opt/venv/bin/python3 02_inspect_pt_zip.py

# Lab 03：mangle 机制 + 折叠 IR 反解析代码噪音的小工具 —— 对应第 03、04 章
/opt/venv/bin/python3 03_module_hierarchy_and_mangling.py

# Lab 04：checkpoint state_dict 的键名结构 —— 对应第 02 章、jit2model_guide.md Step 6
/opt/venv/bin/python3 04_state_dict_vs_code.py
```

## 5.3 每个 lab 看完之后，应该能自己回答的问题

跑完不是目的，能不能**独立复述**才是检验标准。建议对着下面的问题自查一遍，
答不上来的就回对应章节重读。

**Lab 01 跑完之后**：
- 为什么 `torch.jit.trace` 和 `torch.jit.script` 出来的模型，一个能有
  `predict_backbone`/`predict_depth`/`predict_segm` 三个具名方法，另一个不能？
- 如果一个 trace 出来的模型在 batch=5 时 reshape 逻辑正确，能不能保证 batch=3 时也正确？
  为什么"能跑通"不等于"逻辑是动态的"？

**Lab 02 跑完之后**：
- `unified.pt` 这个文件本质上是什么格式？如果不用 `torch.jit.load`，
  能不能用别的工具打开看一眼？能看到什么，不能看到什么？
- `archive/traced_inputs/` 这个目录的存在，说明了这个模型内部的什么事实？

**Lab 03 跑完之后**：
- `___torch_mangle_239` 这种命名里的数字代表什么？两个 mangle 后缀不同的类，
  它们的 Python 源码是不是也不同？
- 给你一份 000_predict_xxx.py，里面有大量 `moduleN = self.module` 和
  `getattr(x, "K")`，你会怎么快速抓出真正的主干逻辑？

**Lab 04 跑完之后**：
- 为什么不能直接把 `torch.load(ckpt_path)` 的结果传给 `model.load_state_dict(...)`？
- checkpoint 里 SE 分支的键名是 `_f.0`/`_f.1`，如果你重建的模型用
  `nn.Sequential(Conv, LeakyReLU, Conv, Sigmoid)` 来实现，键名会变成什么？为什么？

## 5.4 下一步：把这些技巧用在真正的重建任务上

这四章 + 四个 lab 讲完，你应该已经具备了读懂
`jit2engine/jit_code/skill/jit2model.md` 那份操作手册所需要的全部背景知识。
建议的下一步顺序：

1. 回头重读一遍 `jit2model_guide.md`（本仓库已有，路径
   `jit2engine/jit_code/jit2model_guide.md`）和升级版 skill 文档
   `jit2engine/jit_code/skill/jit2model.md`——这次读的时候，
   每遇到一处"这是为什么"，应该都能用前四章的原理去解释，
   而不是死记"规则"。
2. 挑一个还没做过的模型（比如如果 backbone/depth/segmentation 都已经有
   `xxx_model.py`，可以看看 grasp 或其他模型有没有做过），用同样的方法走一遍
   Step 0-8。**这次带着"我知道为什么 IR 反解析代码长这样"的理解去读，
   而不是照着模板抄。**
3. 如果重建过程中遇到"IR 反解析代码和预期不一致"的情况，回到第 04 章的判断框架：
   先分清"这是构造阶段的分支决策（要回源码）"还是"这是被烧录的 Final 常量
   （可以直接读出数值）"，再决定去哪里找答案。

## 5.5 一个进阶练习（可选）：用 `fold_noise` 折叠一份真实的大文件

`jit_learn/labs/03_module_hierarchy_and_mangling.py` 里的 `fold_noise()`
函数目前只处理了两种最常见的噪音模式，是为了教学而写的最小实现。
真实的 `jit2engine/jit_code/depth/000_predict_depth.py` 有 574 行，
噪音密度比 backbone 那 25 行的例子高得多。练习：

```python
import sys
sys.path.insert(0, "/root/brain/brain-service/jit2engine/jit_learn/labs")
from importlib import import_module
lab03 = import_module("03_module_hierarchy_and_mangling")

with open("/root/brain/brain-service/jit2engine/jit_code/depth/000_predict_depth.py") as f:
    raw = f.read()

folded = lab03.fold_noise(raw)
print(folded)
```

跑完看看：折叠之后行数减少了多少？还剩下哪些"折叠不掉"的噪音模式？
（提示：depth 代码里除了 `moduleN`/`getattr` 之外，还有大量
`_all_depthpass` 的多层 `getattr` 嵌套，以及本文没提到的其他辅助变量模式——
如果你想让 `fold_noise` 更强大，可以把新发现的模式加进去，这是一个
"读代码 → 发现新模式 → 扩展工具 → 再读代码更快"的良性循环，
这正是 `jit2model_guide.md` 里所有经验最初被总结出来的方式。）
