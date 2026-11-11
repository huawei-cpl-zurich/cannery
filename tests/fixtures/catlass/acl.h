// CPU-only ACL stand-in: validates bridge argument packing and injected failures.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstddef>
using aclError = int;
static int fault = 0, loads = 0, launches = 0;
using aclrtBinHandle = void*;
using aclrtFuncHandle = void*;
using aclrtArgsHandle = void*;
using aclrtParamHandle = void*;
using aclrtStream = void*;
using aclrtContext = void*;
static uintptr_t current_context = 1;
extern "C" void test_context(int id) { current_context = id; }
static aclError aclrtGetCurrentContext(aclrtContext* context) {
    if (fault == 7) return 107;
    *context = reinterpret_cast<void*>(current_context);
    return 0;
}
constexpr int ACL_RT_BINARY_LOAD_OPT_MAGIC = 2;
constexpr uint32_t ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE = 0x41415246;
struct aclrtBinaryLoadOption { int type; union { uint32_t magic; } value; };
struct aclrtBinaryLoadOptions { aclrtBinaryLoadOption* options; size_t numOpt; };
struct aclrtLaunchKernelCfg { int unused; };
struct Arguments { void* pointers[2]; };
static Arguments payload;
extern "C" void test_fault(int stage) { fault = stage; }
extern "C" int test_loads() { return loads; }
extern "C" int test_launches() { return launches; }
static aclError aclrtBinaryLoadFromData(const void* data, size_t size,
                                       const aclrtBinaryLoadOptions* opts,
                                       aclrtBinHandle* binary) {
    ++loads;
    if (fault == 1) return 101;
    if (size < 64 || opts->numOpt != 1 || opts->options->type != 2 ||
        opts->options->value.magic != ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE) return 199;
    *binary = reinterpret_cast<void*>(uintptr_t(static_cast<const unsigned char*>(data)[8]));
    return 0;
}
static aclError aclrtBinaryGetFunction(aclrtBinHandle binary, const char* symbol,
                                      aclrtFuncHandle* function) {
    if (fault == 2) return 102;
    if (std::strcmp(symbol, "forward")) return 199;
    *function = binary;
    return 0;
}
static aclError aclrtKernelArgsInit(aclrtFuncHandle function, aclrtArgsHandle* args) {
    if (fault == 3) return 103;
    *args = &payload;
    return 0;
}
static aclError aclrtKernelArgsAppend(aclrtArgsHandle args, void* source, size_t size,
                                     aclrtParamHandle* parameter) {
    if (fault == 4) return 104;
    if (args != &payload || size != 16) return 199;
    std::memcpy(payload.pointers, source, size);
    *parameter = nullptr;
    return 0;
}
static aclError aclrtKernelArgsFinalize(aclrtArgsHandle args) {
    return fault == 5 ? 105 : 0;
}
static aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle function, uint32_t blocks,
                                           aclrtStream stream, aclrtLaunchKernelCfg* config,
                                           aclrtArgsHandle args, void* reserved) {
    if (fault == 6) return 106;
    if (blocks != 1 || stream != reinterpret_cast<void*>(99) || !config || reserved ||
        args != &payload) return 199;
    ++launches;
    auto* x = static_cast<float*>(payload.pointers[0]);
    auto* y = static_cast<float*>(payload.pointers[1]);
    for (uintptr_t i = 0; i < reinterpret_cast<uintptr_t>(function); ++i) y[i] = x[i] + 1;
    return 0;
}
