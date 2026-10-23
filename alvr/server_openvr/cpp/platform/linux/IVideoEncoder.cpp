#include "IVideoEncoder.h"
#include "EncodePipeline.h"

namespace alvr {

std::unique_ptr<IVideoEncoder> IVideoEncoder::Create(
    alvr::VkContext& vkCtx,
    const std::string& devicePath,
    const render::Output& output,
    uint32_t width,
    uint32_t height
) {
    // Legacy FFmpeg fallback (VAAPI / NvEnc). In Step 3, VulkanVideoEncoder will be instantiated here first.
    return EncodePipeline::Create(vkCtx, devicePath, output, width, height);
}

} // namespace alvr

