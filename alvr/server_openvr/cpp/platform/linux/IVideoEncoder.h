#pragma once

#include "alvr_server/bindings.h"
#include <cstdint>
#include <memory>
#include <string>

namespace alvr {

class VkContext;

namespace render {
    struct Output;
}

struct FramePacket {
    uint8_t* data = nullptr;
    int size = 0;
    uint64_t pts = 0;
    bool isIDR = false;
};

class IVideoEncoder {
public:
    struct Timestamp {
        uint64_t gpu = 0;
        uint64_t cpu = 0;
    };

    virtual ~IVideoEncoder() = default;

    virtual void PushFrame(uint64_t targetTimestampNs, bool idr) = 0;
    virtual bool GetEncoded(FramePacket& packet) = 0;
    virtual Timestamp GetTimestamp() const { return {}; }
    virtual int GetCodec() const = 0;
    virtual void SetParams(FfiDynamicEncoderParams params) = 0;

    static std::unique_ptr<IVideoEncoder> Create(
        alvr::VkContext& vkCtx,
        const std::string& devicePath,
        const render::Output& output,
        uint32_t width,
        uint32_t height
    );
};

} // namespace alvr

