# TorchScript (JIT) 学习笔记 — 以 unified.pt 为教材

这套材料的目的：让你**看懂** `jit2engine/jit_model/unified.pt/e779c45986c16bb78c851fb56a38c1a9-15`
这个文件到底是什么、为什么长这个样子、以及为什么现有的
`jit2engine/jit_code/skill/jit2model.md` 那套「读 JIT 代码 → 读 handler 源码 → 重建 nn.Module」
的工作流是必须的，而不是绕远路。

## 和已有文档的关系

`jit2engine/jit_code/skill/jit2model.md` 是**操作手册**（怎么做），假设你已经懂 TorchScript。
这套 `jit_learn/` 是**原理课**（为什么），面向"看到 `___torch_mangle_263` 就懵了"的阶段。
读完这里再回去看操作手册，里面很多"坑"会立刻变得可以解释，而不是死记硬背的规则。

## 阅读顺序

| 文件 | 内容 | 对应的真实疑问 |
|------|------|---------------|
| `01_what_is_torchscript.md` | TorchScript 是什么、trace vs script 两种生成方式、为什么要有它 | "这个 .pt 文件为什么不能直接 pickle.load？" |
| `02_anatomy_of_pt_file.md` | 解剖 `unified.pt` 的 zip 结构：code/、data/、data.pkl、mangle | "为什么代码和权重是分开存的？`___torch_mangle_263` 是什么？" |
| `03_reading_generated_code.md` | 逐行读懂 `000_predict_backbone.py` 这种 IR 反解析代码的语法 | "`_00 = getattr(_lateral_fpn_cores, '0')` 这种写法是什么意思？" |
| `04_scripted_vs_python_semantics.md` | TorchScript 子集和普通 Python 的语义差异，为什么源码里的信息会"丢失" | "为什么 JIT 代码看不出激活函数是 LeakyReLU 还是 Identity？" |
| `05_hands_on_lab.md` | 可执行的探索脚本：自己动手 trace/script 一个小模型，对比生成的代码，检查 state_dict | "光看别人分析不够，我要自己动手验证" |
| `06_glossary_and_faq.md` | 术语表 + 常见问题 | 查阅用 |
| `07_pybind11_and_execution_boundary.md` | pybind11 怎么把 C++ 类型包装成 Python 对象（内存布局、trampoline、type_caster）；"在 Python 里执行"和"在 C++ 里执行"精确指什么 | "pybind11 包一层是什么意思？"、"在 Python 里执行 vs 在 C++ 里执行，区别到底是什么？" |

## 配套代码

`jit_learn/labs/` 下是可以直接运行的小实验脚本（不依赖 covariant 框架，纯 PyTorch），
用最小的玩具模型复现 `unified.pt` 里遇到的每一个现象（mangle、trace 常量折叠、
scripted 类型系统限制等），跑起来比看文字理解快得多。

```bash
cd /root/brain/brain-service/jit2engine/jit_learn/labs
python 01_trace_vs_script.py
python 02_inspect_pt_zip.py
python 03_module_hierarchy_and_mangling.py
python 04_state_dict_vs_code.py
python 07_scriptmodule_runtime.py   # ScriptModule 的 C++ 本质 + Python->C++ 调用边界（对应 02 章 2.5b 节）
```

## 一句话总结（先剧透，后面每章都在展开这句话）

> TorchScript 模型文件 = 一份**独立于 Python 源码的、序列化的计算图/字节码**（`code/`）
> + 一份**参数张量的存档**（`data/`），两者通过模块层级（`self._xxx`）互相引用；
> 反解析出来的 `.code` 只保留了"调用了哪些子模块、做了哪些张量操作"，
> **不保留任何在 trace/script 时已经被固定为常量的 Python 值**（比如
> `if self.use_leaky_relu: ...` 这种分支，一旦被决定，分支条件本身就从代码里消失了，
> 只剩下被选中的那条路径）。这就是为什么你必须回源码 + 配置文件才能补全超参数。
