"""
Lab 06 · unified.pt 里 archive/ 下每一种条目，具体长什么样

跑法：
    /opt/venv/bin/python3 06_archive_entries_raw_view.py

对应文档：jit_learn/02_anatomy_of_pt_file.md §2.1

只用 zipfile + pickletools + struct 标准库，不需要 torch、不需要 covariant 环境。
02_inspect_pt_zip.py 已经数过这些条目"有多少个、多大"，这个 lab 换一个角度——
"眼见为实"地把每一类条目的原始内容摊开看一遍：

    archive/data.pkl               模块树骨架，pickle opcode 层面长什么样
    archive/code/*.py              挑一个最短的、不带类包裹的模块级函数完整看一遍
    archive/constants.pkl + constants/0..11   trace 固化的非 Tensor 常量
    archive/traced_inputs.pkl + traced_inputs/0..13   trace 留下的样例输入记录
    archive/byteorder              张量存储的字节序
    archive/.data/version          TorchScript 序列化格式版本号
    archive/.data/serialization_id 序列化时生成的唯一 ID
    archive/data/<N>                纯数值 tensor storage，没有 shape/dtype 头
"""

import io
import pickletools
import struct
import zipfile

PT_PATH = "/root/brain/brain-service/jit2engine/jit_model/unified.pt/e779c45986c16bb78c851fb56a38c1a9-15"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def dis_head(raw: bytes, n_lines: int = 40) -> None:
    """pickletools.dis() 只反汇编 pickle opcode，不实例化任何对象——
    对付 data.pkl 这种引用了 covariant.* 自定义类的 pickle 是安全的。"""
    buf = io.StringIO()
    # pickletools.dis() 干的事，就是把 data.pkl 里那一串二进制字节，按 pickle 协议的规则,翻译成人能读的"一条指令一行"的形式
    pickletools.dis(raw, out=buf)
    lines = buf.getvalue().splitlines()
    print(f"（全部共 {len(lines)} 行 opcode，这里只截取前 {n_lines} 行）")
    for line in lines[:n_lines]:
        print(" ", line)


def _resolve_constant_dtypes(raw: bytes) -> dict:
    """逐条追踪 pickle 的 memo 表（BINPUT/BINGET），而不是只看第一次出现的
    GLOBAL 'torch ...Storage'——pickle 会用 BINGET 复用之前 memo 过的类引用，
    离某个 storage key 最近的 GLOBAL 未必是它实际用的那个类。"""
    memo = {}
    pending_class = None
    key_dtype = {}
    for opcode, arg, _pos in pickletools.genops(raw):
        if opcode.name == "GLOBAL":
            pending_class = arg
        elif opcode.name in ("BINPUT", "PUT", "LONG_BINPUT"):
            memo[arg] = pending_class
        elif opcode.name in ("BINGET", "GET", "LONG_BINGET"):
            pending_class = memo.get(arg, pending_class)
        elif opcode.name in ("BINUNICODE", "SHORT_BINUNICODE"):
            if isinstance(arg, str) and arg.isdigit() and pending_class and "Storage" in pending_class:
                key_dtype[arg] = "int64" if "Long" in pending_class else "float64"
    return key_dtype


def main() -> None:
    z = zipfile.ZipFile(PT_PATH)

    section("1. archive/data.pkl —— 模块树骨架，pickle opcode 层面长什么样")
    raw = z.read("archive/data.pkl")
    print(f"原始字节数: {len(raw)}")
    print("\n前 200 字节（原始 bytes，能看到里面混着类路径字符串）：")
    print(raw[:200])
    print("\npickletools.dis() 反汇编：")
    dis_head(raw, 40)
    print(
        "\n观察：一开头就是 GLOBAL '__torch__.covariant.model.utils.export.core\n"
        "WrapperModule'，紧接着 NEWOBJ + EMPTY_DICT + 一串 BINUNICODE 属性名\n"
        "('training'/'module'/'_backbone_fpn_handler'...) ——pickle 协议把\n"
        "'这是什么类、有哪些属性、属性又是什么类'编码成 GLOBAL/NEWOBJ 这类 opcode\n"
        "引用，不需要真的执行这些引用（那需要 covariant 环境），单看 opcode 序列\n"
        "就能看到模块树是怎么一层套一层展开的——这正是'骨架'的字面意思。"
    )

    section("2. archive/code/*.py —— 挑一个最短的、不带类包裹的模块级函数完整看一遍")
    code_path = "archive/code/__torch__/covariant/common/torch_ops/roi_upsample.py"
    code_text = z.read(code_path).decode()
    print(f"文件: {code_path}  ({len(code_text)} 字节)\n")
    print(code_text)
    print(
        "对应 02_anatomy_of_pt_file.md §2.2 说的'也可能根本不是类，是一个模块级函数'——\n"
        "这份文件里没有 class，直接是顶层 def，因为它在源码里本来就是普通函数，\n"
        "不是某个 nn.Module 的方法。"
    )

    section("3. archive/constants.pkl + archive/constants/0..11 —— trace/script 固化的字面常量")
    craw = z.read("archive/constants.pkl")
    print(f"constants.pkl 原始字节数: {len(craw)}")
    dis_head(craw, 30)
    print(
        "\n注意：不能只看 opcode 里第一次出现的 GLOBAL 'torch LongStorage'/\n"
        "'torch DoubleStorage'就断定'后面全是同一种 dtype'——pickle 会用\n"
        "BINGET 复用之前 BINPUT 过的类引用（memo），第 10 个键实际复用的是\n"
        "LongStorage 的memo 槽，不是离它最近的那个 DoubleStorage GLOBAL。\n"
        "用 pickletools.genops() 逐条追踪 memo 状态才能拿到每个 key 真实的 dtype："
    )
    dtypes = _resolve_constant_dtypes(craw)
    for i in range(12):
        b = z.read(f"archive/constants/{i}")
        dtype = dtypes.get(str(i), "float64")
        value = struct.unpack("<q", b)[0] if dtype == "int64" else struct.unpack("<d", b)[0]
        print(f"  constants/{i:<2d} raw={b.hex():<18} dtype={dtype:6s} value={value!r}")
    print(
        "\n现在这 12 个数全部落在合理量级：0/1/2/4/255 明显是整数,1e-08/1e-05\n"
        "是常见 eps,0.5/10.0/5.0 是常见系数,3.14159...是 pi。再去 code/*.py 里\n"
        "搜 'CONSTANTS.c<N>' 能直接找到它们被用在哪一行——比如\n"
        "regnet_handler.py 里 'torch.sub(torch.mul(x, CONSTANTS.c10), _pixel_means)'，\n"
        "c10=255，这是'把 [0,1] 归一化的像素值乘回 255 再减均值'的图像预处理；\n"
        "positional_encoding_handler.py 里 'torch.mul(_8, CONSTANTS.c11)'，\n"
        "c11=pi，是位置编码常见的正弦/余弦项里的 2π 因子；\n"
        "normals_handler.py 里 'torch.add(_27, CONSTANTS.c1)'，c1=1e-08，\n"
        "是防止除零的数值稳定性 eps。"
    )
    print(
        "\n这就是'trace/script 过程中固化下来的常量（非 Tensor）'的真实含义：\n"
        "源码里写死的 Python 字面量（0.5、255、pi、1e-8……），forward 执行时\n"
        "需要拿到一个真正的 Tensor 去参与运算（torch.mul/torch.add 的参数得是\n"
        "Tensor），但它们不是 nn.Parameter/buffer,不属于模型要学习或要在\n"
        "state_dict 里出现的权重——所以既不会被塞进 data.pkl 的模块属性树,\n"
        "也不会像 kernel_size 那样直接印成 Final 字面量（因为它们是 forward\n"
        "方法体里的字面量，不是构造参数），而是单独开一个'常量表'\n"
        "（constants.pkl + constants/<N>），.code 里用 CONSTANTS.c<N> 这个特殊\n"
        "名字来引用第 N 个常量。'非 Tensor'说的是'不是模型要学习的权重张量'，\n"
        "不是说存储机制上不用 Tensor——机制上它们照样是靠\n"
        "torch._utils._rebuild_tensor_v2 包出来的（0 维/1 元素）Tensor 对象，\n"
        "只是被单独放进了一张和 data.pkl/data/ 平行的常量表里。"
    )

    section("4. archive/traced_inputs.pkl + archive/traced_inputs/0..13 —— trace 留下的样例输入")
    traw = z.read("archive/traced_inputs.pkl")
    print(f"traced_inputs.pkl 原始字节数: {len(traw)}")
    dis_head(traw, 30)
    print("\narchive/traced_inputs/0..13 各自的文件大小：")
    for i in range(14):
        info = z.getinfo(f"archive/traced_inputs/{i}")
        print(f"  traced_inputs/{i:<2d} {info.file_size} bytes")
    print(
        "\n矛盾点：pickle opcode 里 key '0' 声明的是一个 HalfStorage、\n"
        "device='cuda:0'、size=2162688 个元素（key '1' 是 LongStorage size=2，\n"
        "key '7' 是 FloatStorage size=1）——也就是说 trace 时记录的样例输入本来\n"
        "是有真实尺寸的（2162688 个 half ≈ 4.2MB），但对应的\n"
        "archive/traced_inputs/0 文件实际只有 0 字节。说明导出流程里,'样例输入的\n"
        "形状/dtype/device 元数据'被保留了下来,但'真实数值'在最终 save 之前被\n"
        "清空/置零了——大概率是导出脚本刻意做的体积优化（推理完全不需要这些\n"
        "调试用的样例值,留着只会白白增加文件大小），不是 bug。这也是为什么\n"
        "'archive/traced_inputs/<N> 这个文件本身存在'才是 trace 的物证，而不是\n"
        "'里面有非零字节'——文件存在 = 当时确实调用过 trace_module 并记录了\n"
        "输入签名,至于数值有没有被后续步骤清空是另一件事。"
    )

    section("5. archive/byteorder / archive/.data/version / archive/.data/serialization_id")
    print("byteorder:        ", z.read("archive/byteorder"))
    print("version:          ", z.read("archive/.data/version"))
    print("serialization_id: ", z.read("archive/.data/serialization_id"))
    print(
        "\n都是纯文本：byteorder 声明 data/ 下所有 tensor storage 按小端存的字节；\n"
        "version 是 TorchScript zip 序列化格式的版本号；serialization_id 是这次\n"
        "save 生成的一个随机唯一 ID（两次 save 同一个模型，这串数字也会不同）。"
    )

    section("6. archive/data/<N> —— 纯数值 tensor storage，没有 shape/dtype 头")
    n = "archive/data/1"
    raw = z.read(n)
    print(f"{n} ({len(raw)} bytes) 原始十六进制：\n  {raw.hex()}")
    print("\n如果不知道 dtype，随手当 float32 解一遍（错误示范）：")
    print(" ", struct.unpack(f"<{len(raw)//4}f", raw)[:6], "...")
    print(
        "这些数量级 (1e-9 ~ 1e-17) 明显不是正常的卷积权重——因为这份 unified.pt\n"
        "是 fp16 推理模型（02_anatomy_of_pt_file.md §2.4 提到的那个 fp32/fp16 两倍\n"
        "关系），64 字节实际是 32 个 fp16 数，不是 16 个 fp32 数。按 fp16 (struct 'e')\n"
        "重新解一遍："
    )
    print(" ", struct.unpack(f"<{len(raw)//2}e", raw)[:8], "...")
    print(
        "\n数量级立刻回到正常权重范围 (±0.5 左右)。这就是文档反复强调的\n"
        "'archive/data/<N> 没有 shape/dtype 信息'——同一段裸字节，按不同 dtype\n"
        "解读出来的数字完全不同，真正的 dtype/shape/stride 只存在 data.pkl 里，\n"
        "脱离 data.pkl 单看这个文件，连'这是几个数'都不知道。"
    )


if __name__ == "__main__":
    main()
