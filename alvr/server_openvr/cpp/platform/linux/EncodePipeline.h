#pragma once

#include "IVideoEncoder.h"
#include "Renderer.hpp"
#include "alvr_server/bindings.h"
#include <cstdint>
#include <memory>

#include "VkContext.hpp"

extern "C" struct AVCodecContext;
extern "C" struct AVPacket;

namespace alvr {

class EncodePipeline : public IVideoEncoder {
public:
    ~EncodePipeline() override;

    bool EncodeFrame(uint64_t targetTimestampNs, bool idr, FramePacket& packet) override;
    virtual void PushFrame(uint64_t targetTimestampNs, bool idr) = 0;
    virtual bool GetEncoded(FramePacket& data);

    Timestamp GetTimestamp() const override { return timestamp; }
    int GetCodec() const override;

    void SetParams(FfiDynamicEncoderParams params) override;
    static std::unique_ptr<IVideoEncoder> Create(
        alvr::VkContext& vk_ctx,
        const std::string& devicePath,
        const render::Output& output,
        uint32_t width,
        uint32_t height
    );

protected:
    AVCodecContext* encoder_ctx = nullptr; // shall be initialized by child class
    AVPacket* encoder_packet = nullptr;
    Timestamp timestamp = {};
};

} // namespace alvr
