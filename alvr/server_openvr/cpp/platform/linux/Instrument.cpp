#include "Instrument.h"

#include "alvr_server/Logger.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <pwd.h>
#include <unistd.h>

namespace alvr::instrument {

// -------------------------------------------------------------------
// Internal state
// -------------------------------------------------------------------

static FILE* g_file = nullptr;
static std::mutex g_fileMutex;
static std::chrono::steady_clock::time_point g_initTime;

// Experiment overrides, read once at Init().
struct Overrides {
    bool formatActive = false;
    uint32_t format = 0;

    bool usageFlagsActive = false;
    uint32_t usageFlags = 0;

    bool renderableActive = false;
    bool renderable = false;

    bool mappableActive = false;
    bool mappable = false;

    bool computeAccessActive = false;
    bool computeAccess = false;
};
static Overrides g_overrides;

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

static int64_t ElapsedUs() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - g_initTime).count();
}

static bool ReadEnvBool(const char* name, bool* out) {
    const char* val = getenv(name);
    if (!val) return false;
    *out = (val[0] == '1');
    return true;
}

static bool ReadEnvU32(const char* name, uint32_t* out) {
    const char* val = getenv(name);
    if (!val) return false;
    char* end = nullptr;
    unsigned long v = strtoul(val, &end, 0);
    if (end == val) return false;
    *out = static_cast<uint32_t>(v);
    return true;
}

// Minimal JSON-safe string escape for fixed fields. We only emit
// known keys and numeric values, so this is a safety net, not a
// general-purpose encoder.
static void WriteJsonString(FILE* f, const char* s) {
    fputc('"', f);
    for (; *s; ++s) {
        switch (*s) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f); break;
        default:   fputc(*s, f); break;
        }
    }
    fputc('"', f);
}

// -------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------

void Init() {
    g_initTime = std::chrono::steady_clock::now();

    // Build path: ~/alvr_compositor_instrument.jsonl
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        if (pw) home = pw->pw_dir;
    }
    if (!home) home = "/tmp";

    char path[512];
    snprintf(path, sizeof(path), "%s/alvr_compositor_instrument.jsonl", home);

    g_file = fopen(path, "a");
    if (!g_file) {
        Info("Instrument: failed to open %s for writing\n", path);
        return;
    }
    Info("Instrument: logging to %s\n", path);

    // Read experiment overrides
    g_overrides.formatActive = ReadEnvU32("ALVR_INSTR_FORMAT", &g_overrides.format);
    g_overrides.usageFlagsActive = ReadEnvU32("ALVR_INSTR_USAGE_FLAGS", &g_overrides.usageFlags);
    g_overrides.renderableActive = ReadEnvBool("ALVR_INSTR_RENDERABLE", &g_overrides.renderable);
    g_overrides.mappableActive = ReadEnvBool("ALVR_INSTR_MAPPABLE", &g_overrides.mappable);
    g_overrides.computeAccessActive = ReadEnvBool("ALVR_INSTR_COMPUTE_ACCESS", &g_overrides.computeAccess);

    // Log active overrides
    if (g_overrides.formatActive)
        Info("Instrument: ALVR_INSTR_FORMAT override active: %u\n", g_overrides.format);
    if (g_overrides.usageFlagsActive)
        Info("Instrument: ALVR_INSTR_USAGE_FLAGS override active: 0x%x\n", g_overrides.usageFlags);
    if (g_overrides.renderableActive)
        Info("Instrument: ALVR_INSTR_RENDERABLE override active: %d\n", g_overrides.renderable);
    if (g_overrides.mappableActive)
        Info("Instrument: ALVR_INSTR_MAPPABLE override active: %d\n", g_overrides.mappable);
    if (g_overrides.computeAccessActive)
        Info("Instrument: ALVR_INSTR_COMPUTE_ACCESS override active: %d\n", g_overrides.computeAccess);

    // Session start marker
    {
        std::lock_guard<std::mutex> lock(g_fileMutex);
        fprintf(g_file,
            "{\"type\":\"session_start\",\"ts_us\":0,\"pid\":%d}\n",
            (int)getpid());
        fflush(g_file);
    }
}

void Shutdown() {
    if (g_file) {
        std::lock_guard<std::mutex> lock(g_fileMutex);
        fprintf(g_file,
            "{\"type\":\"session_end\",\"ts_us\":%lld}\n",
            (long long)ElapsedUs());
        fflush(g_file);
        fclose(g_file);
        g_file = nullptr;
    }
}

void LogCreateSwapTextureSet(
    uint32_t pid,
    uint32_t requestedFormat, uint32_t effectiveFormat,
    uint32_t width, uint32_t height, uint32_t sampleCount,
    uint32_t requestedUsageFlags, uint32_t effectiveUsageFlags,
    bool requestedRenderable, bool effectiveRenderable,
    bool requestedMappable, bool effectiveMappable,
    bool requestedComputeAccess, bool effectiveComputeAccess,
    const uint64_t sharedHandles[3], const int fds[3],
    bool success
) {
    if (!g_file) return;
    std::lock_guard<std::mutex> lock(g_fileMutex);
    fprintf(g_file,
        "{\"type\":\"create_swap_texture_set\","
        "\"ts_us\":%lld,"
        "\"pid\":%u,"
        "\"req_format\":%u,\"eff_format\":%u,"
        "\"width\":%u,\"height\":%u,\"sample_count\":%u,"
        "\"req_usage_flags\":%u,\"eff_usage_flags\":%u,"
        "\"req_renderable\":%s,\"eff_renderable\":%s,"
        "\"req_mappable\":%s,\"eff_mappable\":%s,"
        "\"req_compute_access\":%s,\"eff_compute_access\":%s,"
        "\"handles\":[%llu,%llu,%llu],"
        "\"fds\":[%d,%d,%d],"
        "\"success\":%s}\n",
        (long long)ElapsedUs(),
        pid,
        requestedFormat, effectiveFormat,
        width, height, sampleCount,
        requestedUsageFlags, effectiveUsageFlags,
        requestedRenderable ? "true" : "false",
        effectiveRenderable ? "true" : "false",
        requestedMappable ? "true" : "false",
        effectiveMappable ? "true" : "false",
        requestedComputeAccess ? "true" : "false",
        effectiveComputeAccess ? "true" : "false",
        (unsigned long long)sharedHandles[0],
        (unsigned long long)sharedHandles[1],
        (unsigned long long)sharedHandles[2],
        fds[0], fds[1], fds[2],
        success ? "true" : "false");
    fflush(g_file);
}

void LogDestroySwapTextureSet(uint64_t handle, uint32_t pid, double lifetimeSec) {
    if (!g_file) return;
    std::lock_guard<std::mutex> lock(g_fileMutex);
    fprintf(g_file,
        "{\"type\":\"destroy_swap_texture_set\","
        "\"ts_us\":%lld,"
        "\"handle\":%llu,"
        "\"pid\":%u,"
        "\"lifetime_sec\":%.6f}\n",
        (long long)ElapsedUs(),
        (unsigned long long)handle,
        pid,
        lifetimeSec);
    fflush(g_file);
}

void LogDestroyAllSwapTextureSets(uint32_t pid, uint32_t count) {
    if (!g_file) return;
    std::lock_guard<std::mutex> lock(g_fileMutex);
    fprintf(g_file,
        "{\"type\":\"destroy_all_swap_texture_sets\","
        "\"ts_us\":%lld,"
        "\"pid\":%u,"
        "\"count\":%u}\n",
        (long long)ElapsedUs(),
        pid,
        count);
    fflush(g_file);
}

void LogGetNextSwapTextureSetIndex(
    const uint64_t handles[2], const uint32_t indices[2]
) {
    if (!g_file) return;
    std::lock_guard<std::mutex> lock(g_fileMutex);
    fprintf(g_file,
        "{\"type\":\"get_next_index\","
        "\"ts_us\":%lld,"
        "\"handles\":[%llu,%llu],"
        "\"indices\":[%u,%u]}\n",
        (long long)ElapsedUs(),
        (unsigned long long)handles[0],
        (unsigned long long)handles[1],
        indices[0], indices[1]);
    fflush(g_file);
}

void LogPresent(
    uint32_t leftIdx, uint32_t rightIdx,
    uint64_t targetTimestampNs,
    uint64_t syncTextureHandle,
    const int syncFds[6], const bool fencePending[6],
    bool isFirstPresent
) {
    if (!g_file) return;
    std::lock_guard<std::mutex> lock(g_fileMutex);
    fprintf(g_file,
        "{\"type\":\"present\","
        "\"ts_us\":%lld,"
        "\"left_idx\":%u,\"right_idx\":%u,"
        "\"target_ts_ns\":%llu,"
        "\"sync_texture\":%llu,"
        "\"sync_fds\":[%d,%d,%d,%d,%d,%d],"
        "\"fence_pending\":[%s,%s,%s,%s,%s,%s],"
        "\"first_present\":%s}\n",
        (long long)ElapsedUs(),
        leftIdx, rightIdx,
        (unsigned long long)targetTimestampNs,
        (unsigned long long)syncTextureHandle,
        syncFds[0], syncFds[1], syncFds[2],
        syncFds[3], syncFds[4], syncFds[5],
        fencePending[0] ? "true" : "false",
        fencePending[1] ? "true" : "false",
        fencePending[2] ? "true" : "false",
        fencePending[3] ? "true" : "false",
        fencePending[4] ? "true" : "false",
        fencePending[5] ? "true" : "false",
        isFirstPresent ? "true" : "false");
    fflush(g_file);
}

bool OverrideFormat(uint32_t* out) {
    if (!g_overrides.formatActive) return false;
    *out = g_overrides.format;
    return true;
}

bool OverrideUsageFlags(uint32_t* out) {
    if (!g_overrides.usageFlagsActive) return false;
    *out = g_overrides.usageFlags;
    return true;
}

bool OverrideRenderable(bool* out) {
    if (!g_overrides.renderableActive) return false;
    *out = g_overrides.renderable;
    return true;
}

bool OverrideMappable(bool* out) {
    if (!g_overrides.mappableActive) return false;
    *out = g_overrides.mappable;
    return true;
}

bool OverrideComputeAccess(bool* out) {
    if (!g_overrides.computeAccessActive) return false;
    *out = g_overrides.computeAccess;
    return true;
}

} // namespace alvr::instrument
