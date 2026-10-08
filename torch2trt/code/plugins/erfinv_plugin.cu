// ErfInv TensorRT plugin (IPluginV3), the "write a plugin" answer that the
// ONNX->TRT route forces on you when the parser meets an op it does not know.
//
// Chapter 05 / 10: the customop model contains torch.erfinv. After the ONNX
// side emits a `custom::ErfInv` node (see exp_custom_plugin.py), TensorRT's
// ONNX parser does not recognise the op type, so it looks up the plugin
// registry for a creator named "ErfInv" version "1". This file provides it.
//
// Build:  bash plugins/build.sh   ->  artifacts/liberfinv_plugin.so
//
// The amount of code here is the point. Everything below except `erfinvKernel`
// is interface boilerplate: shape inference, dtype/format negotiation,
// serialization, cloning, a creator. That is the fixed cost of a TRT plugin,
// and it is the same whether the kernel is one line or one thousand.

#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace erfinv_impl
{
constexpr char const* kNAME = "ErfInv";
constexpr char const* kVERSION = "1";
constexpr char const* kNAMESPACE = "";

template <typename T>
__global__ void erfinvKernel(T const* __restrict__ in, T* __restrict__ out, int64_t n)
{
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < n)
    {
        // compute in fp32 even for half I/O: erfinv blows up near +-1 and
        // fp16 arithmetic there is too coarse
        out[i] = static_cast<T>(erfinvf(static_cast<float>(in[i])));
    }
}

int64_t volume(Dims const& d)
{
    int64_t v = 1;
    for (int32_t i = 0; i < d.nbDims; ++i)
        v *= d.d[i];
    return v;
}
} // namespace erfinv_impl
using namespace erfinv_impl;

class ErfInvPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    // ---- IPluginV3 -------------------------------------------------------
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new ErfInvPlugin(*this); }

    // ---- IPluginV3OneCore ------------------------------------------------
    char const* getPluginName() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    char const* getPluginNamespace() const noexcept override { return kNAMESPACE; }

    // ---- IPluginV3OneBuild -----------------------------------------------
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t, DataType const* inputTypes, int32_t) const noexcept override
    {
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t,
        IExprBuilder&) noexcept override
    {
        outputs[0] = inputs[0]; // elementwise: shape in == shape out, dynamic dims included
        return 0;
    }

    // TensorRT asks this for every (position, dtype, format) it is considering.
    // Accepting both fp32 and fp16 lets the builder keep the surrounding layers
    // in fp16 without inserting casts around the plugin.
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t, int32_t) noexcept override
    {
        auto const& d = inOut[pos].desc;
        bool ok = d.format == TensorFormat::kLINEAR && (d.type == DataType::kFLOAT || d.type == DataType::kHALF);
        if (pos == 1)
            ok = ok && d.type == inOut[0].desc.type;
        return ok;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override
    {
        return 0;
    }

    // ---- IPluginV3OneRuntime ---------------------------------------------
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        int64_t n = volume(inputDesc[0].dims);
        int32_t block = 256;
        int64_t grid = (n + block - 1) / block;
        if (inputDesc[0].type == DataType::kHALF)
            erfinvKernel<__half><<<grid, block, 0, stream>>>(
                static_cast<__half const*>(inputs[0]), static_cast<__half*>(outputs[0]), n);
        else
            erfinvKernel<float><<<grid, block, 0, stream>>>(
                static_cast<float const*>(inputs[0]), static_cast<float*>(outputs[0]), n);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

    // No attributes to persist; a plugin with parameters would serialize them
    // here and the creator would read them back in createPlugin().
    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
        return &mFC;
    }

private:
    PluginFieldCollection mFC{};
};

class ErfInvPluginCreator : public IPluginCreatorV3One
{
public:
    ErfInvPluginCreator()
    {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
    }
    char const* getPluginName() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    char const* getPluginNamespace() const noexcept override { return kNAMESPACE; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }

    IPluginV3* createPlugin(char const*, PluginFieldCollection const*, TensorRTPhase) noexcept override
    {
        return new ErfInvPlugin();
    }

private:
    PluginFieldCollection mFC{};
};

// Static registration: the creator lands in the global plugin registry the
// moment the .so is loaded (ctypes.CDLL, trtexec --staticPlugins, or dlopen in
// a C++ server). No registration code is needed on the caller's side.
REGISTER_TENSORRT_PLUGIN(ErfInvPluginCreator);
