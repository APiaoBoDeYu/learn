"""
Lab 02 · 解剖 unified.pt 这个 zip 文件

跑法：
    /opt/venv/bin/python3 02_inspect_pt_zip.py

对应文档：jit_learn/02_anatomy_of_pt_file.md

不需要 torch，不需要 GPU，纯 zipfile 标准库就能看完这个文件的骨架。
"""
import zipfile
from collections import Counter

PT_PATH = "/root/brain/brain-service/jit2engine/jit_model/unified.pt/e779c45986c16bb78c851fb56a38c1a9-15"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    z = zipfile.ZipFile(PT_PATH)
    names = z.namelist()

    section("1. 总览：这个 zip 里有多少文件，分几类")
    print(f"总文件数: {len(names)}")

    top_level_dirs = Counter(n.split("/")[1] if n.count("/") >= 1 else n for n in names)
    print("archive/ 下第一层子目录/文件及各自条目数:")
    for k, v in sorted(top_level_dirs.items()):
        print(f"  {k:20s} {v:5d} 个条目")

    section("2. code/ 目录：每个'类'一个 IR 反解析代码文件")
    code_py = [n for n in names if n.startswith("archive/code/") and n.endswith(".py")]
    mangled = [n for n in code_py if "___torch_mangle_" in n]
    plain = [n for n in code_py if "___torch_mangle_" not in n]
    print(f"code/ 下 .py 文件总数: {len(code_py)}")
    print(f"  其中带 ___torch_mangle_ 后缀（同一个类被编译出多份不同版本): {len(mangled)}")
    print(f"  不带 mangle 后缀（该类在整个模型里只被编译了一次): {len(plain)}")
    print("\n举例 - 不带 mangle 的（说明这个类在模型里的所有实例编译结果完全一样）:")
    for n in plain[:5]:
        print(f"  {n}")
    print("\n举例 - 带 mangle 的（同一个 Python 类，因为不同实例的超参不同，被编译出多份独立版本）:")
    for n in mangled[:5]:
        print(f"  {n}")

    section("3. 按'去掉 mangle 后缀'归类，看哪些类被实例化了最多次")
    def base_name(n: str) -> str:
        # archive/code/__torch__/torch/nn/modules/conv/___torch_mangle_239.py
        #   -> archive/code/__torch__/torch/nn/modules/conv (去掉文件名，只留目录路径当"类家族"标识)
        parts = n.rsplit("/", 1)
        return parts[0] if len(parts) == 2 else n

    family_counts = Counter(base_name(n) for n in code_py)
    print("被编译出最多个'版本'的类家族 top 10（数字越大，说明这个基础组件在模型里被大量复用、\n"
          "但因为构造参数各不相同，TorchScript 编译器把每种参数组合都当成独立版本存了一份）:")
    for family, cnt in family_counts.most_common(10):
        short = family.replace("archive/code/__torch__/", "")
        print(f"  {cnt:4d}  {short}")

    section("4. data/ 目录：纯数值权重存储")
    data_files = [n for n in names if n.startswith("archive/data/")]
    print(f"data/ 下条目数（= 模型里 Tensor storage 的总数): {len(data_files)}")
    sizes = sorted(((z.getinfo(n).file_size, n) for n in data_files), reverse=True)
    total_bytes = sum(z.getinfo(n).file_size for n in data_files)
    print(f"所有 data/ 条目总大小: {total_bytes / 1e6:.1f} MB")
    print("最大的 5 个 tensor storage:")
    for sz, n in sizes[:5]:
        print(f"  {sz/1e6:8.2f} MB  {n}")

    section("5. 结构性文件：data.pkl / constants.pkl / traced_inputs.pkl / version")
    for meta_name in ["archive/data.pkl", "archive/constants.pkl", "archive/traced_inputs.pkl",
                       "archive/version", "archive/byteorder"]:
        if meta_name in names:
            info = z.getinfo(meta_name)
            print(f"  {meta_name:35s} {info.file_size:8d} bytes")
        else:
            print(f"  {meta_name:35s} (不存在于这个文件里)")

    section("6. 直接看 data.pkl 的前 300 字节（不反序列化，只看原始 pickle opcode 里的字符串）")
    raw = z.read("archive/data.pkl")
    print(raw[:300])
    print(
        "\n注意看里面出现的 '__torch__.covariant.model.utils.export.core\\nWrapperModule' 这样的\n"
        "类路径字符串——这就是 data.pkl 记录的模块树根节点。pickle 协议会把'这是什么类'\n"
        "编码成这种 REDUCE/GLOBAL opcode 引用的字符串，理论上可以完全不用 torch 就用\n"
        "pickletools.dis() 去看完整的重建指令序列，但实际调试时直接 torch.jit.load 更省事。"
    )

    section("7. 试着（在不装载真实算子的情况下）安全地看一眼有没有自定义 op 引用")
    # 直接在原始字节里找特征字符串，不需要反序列化，用来快速判断这个模型
    # 依赖了哪些自定义 CUDA op —— 对应 extrace_info.py 里 load_library 的那些 .so
    custom_op_markers = [b"soft_bin_ops", b"roi_upsample", b"roi_align", b"patch_outerprod",
                          b"demosaicing", b"sym3eig"]
    print("在 data.pkl 原始字节里查找已知自定义算子名称的出现情况:")
    for marker in custom_op_markers:
        found = marker in raw
        print(f"  {marker.decode():20s} {'出现' if found else '未出现'} 在 data.pkl 里")
    print(
        "\n（是否'出现'取决于该算子是否被这三个 predict_* 方法中的某一个直接引用；\n"
        " 完整判断仍应以 extrace_info.py 里实际 load_library 后 torch.jit.load 成功与否为准，\n"
        " 这里只是演示'不反序列化也能做初步侦察'的技巧。）"
    )


if __name__ == "__main__":
    main()
