#pragma once

#include "alvr_server/bindings.h"
#include <cstdint>
#include <memory>
#include <string>

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

namespace alvr {

class VkContext;

namespace render {
    struct Output;
    struct YuvImage;
}

struct FramePacket {
    uint8_t* data = nullptr;
    int size = 0;
    uint64_t pts = 0;
    bool isIDR = false;
};

struct VideoFrameSlot {
    vk::Image image = VK_NULL_HANDLE;
    vk::ImageView yuvView = VK_NULL_HANDLE;
    vk::ImageView yView = VK_NULL_HANDLE;
    vk::ImageView uvView = VK_NULL_HANDLE;
    vk::Extent2D extent {};
    vk::Format format = vk::Format::eUndefined;
    vk::Semaphore readySemaphore = VK_NULL_HANDLE;
};

class IVideoEncoder {
public:
    struct Timestamp {
        uint64_t gpu = 0;
        uint64_t cpu = 0;
    };

    virtual ~IVideoEncoder() = default;

    virtual void SetParams(FfiDynamicEncoderParams params) = 0;
    virtual int GetCodec() const = 0;
    virtual Timestamp GetTimestamp() const { return {}; }

    // Encodes a frame (1:1 in real-time VR streaming)
    virtual bool EncodeFrame(
        uint64_t targetTimestampNs,
        bool idr,
        FramePacket& packet
    ) = 0;

    // Encodes a frame from a native Vulkan YUV slot with GPU semaphore synchronization
    virtual bool EncodeFrame(
        const VideoFrameSlot& slot,
        uint64_t targetTimestampNs,
        bool idr,
        FramePacket& packet
    ) {
        (void)slot;
        return EncodeFrame(targetTimestampNs, idr, packet);
    }

    static std::unique_ptr<IVideoEncoder> Create(
        alvr::VkContext& vkCtx,
        const std::string& devicePath,
        const render::Output& output,
        uint32_t width,
        uint32_t height
    );
};

} // namespace alvr
