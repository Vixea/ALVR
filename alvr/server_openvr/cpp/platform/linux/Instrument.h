#pragma once

#include <cstdint>

namespace alvr::instrument {

/// Call once at startup (OvrDirectModeComponent constructor).
/// Opens ~/alvr_compositor_instrument.jsonl, reads env vars for
/// experiment overrides.
void Init();

/// Structured JSONL event writers — one per instrumented call site.
void LogCreateSwapTextureSet(
    uint32_t pid,
    uint32_t requestedFormat, uint32_t effectiveFormat,
    uint32_t width, uint32_t height, uint32_t sampleCount,
    uint32_t requestedUsageFlags, uint32_t effectiveUsageFlags,
    bool requestedRenderable, bool effectiveRenderable,
    bool requestedMappable, bool effectiveMappable,
    bool requestedComputeAccess, bool effectiveComputeAccess,
    // outputs after creation
    const uint64_t sharedHandles[3], const int fds[3],
    bool success
);

void LogDestroySwapTextureSet(uint64_t handle, uint32_t pid,
                               double lifetimeSec);
void LogDestroyAllSwapTextureSets(uint32_t pid, uint32_t count);

void LogGetNextSwapTextureSetIndex(
    const uint64_t handles[2], const uint32_t indices[2]
);

void LogPresent(
    uint32_t leftIdx, uint32_t rightIdx,
    uint64_t targetTimestampNs,
    uint64_t syncTextureHandle,
    // fence state for all 6 texture slots
    const int syncFds[6], const bool fencePending[6],
    bool isFirstPresent
);

/// Experiment overrides — return true if an override is active.
/// When active, *out is set to the override value.
bool OverrideFormat(uint32_t* out);
bool OverrideUsageFlags(uint32_t* out);
bool OverrideRenderable(bool* out);
bool OverrideMappable(bool* out);
bool OverrideComputeAccess(bool* out);

void Shutdown();

} // namespace alvr::instrument
