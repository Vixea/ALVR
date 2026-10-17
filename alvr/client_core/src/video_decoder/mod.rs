#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "linux")]
mod linux;

use alvr_common::anyhow::Result;
use alvr_session::{CodecType, MediacodecProperty};
use std::time::Duration;

#[derive(Clone, Default, PartialEq)]
pub struct VideoDecoderConfig {
    pub codec: CodecType,
    pub force_software_decoder: bool,
    pub max_buffering_frames: f32,
    pub buffering_history_weight: f32,
    pub options: Vec<(String, MediacodecProperty)>,
    pub config_buffer: Vec<u8>,
}

pub struct VideoDecoderSink {
    #[cfg(target_os = "android")]
    inner: android::VideoDecoderSink,
    #[cfg(target_os = "linux")]
    inner: linux::VideoDecoderSink,
}

impl VideoDecoderSink {
    // returns true if frame has been successfully enqueued
    #[allow(unused_variables)]
    pub fn push_nal(&mut self, timestamp: Duration, nal: &[u8]) -> bool {
        #[cfg(any(target_os = "android", target_os = "linux"))]
        {
            alvr_common::show_err(self.inner.push_frame_nal(timestamp, nal)).unwrap_or(false)
        }
        #[cfg(not(any(target_os = "android", target_os = "linux")))]
        false
    }
}

pub struct VideoDecoderSource {
    #[cfg(target_os = "android")]
    inner: android::VideoDecoderSource,
    #[cfg(target_os = "linux")]
    inner: linux::VideoDecoderSource,
}

impl VideoDecoderSource {
    /// If a frame is available, return the timestamp and the native frame handle
    /// (`AHardwareBuffer` on Android, `LinuxVideoFrame` on Linux).
    pub fn get_frame(&mut self) -> Option<(Duration, *mut std::ffi::c_void)> {
        #[cfg(any(target_os = "android", target_os = "linux"))]
        {
            self.inner.dequeue_frame()
        }
        #[cfg(not(any(target_os = "android", target_os = "linux")))]
        None
    }
}

// report_frame_decoded: (target_timestamp: Duration) -> ()
#[allow(unused_variables)]
pub fn create_decoder(
    config: VideoDecoderConfig,
    report_frame_decoded: impl Fn(Result<Duration>) + Send + Sync + 'static,
) -> (VideoDecoderSink, VideoDecoderSource) {
    #[cfg(target_os = "android")]
    {
        let (sink, source) = android::video_decoder_split(
            config.clone(),
            config.config_buffer,
            report_frame_decoded,
        )
        .unwrap();

        (
            VideoDecoderSink { inner: sink },
            VideoDecoderSource { inner: source },
        )
    }
    #[cfg(target_os = "linux")]
    {
        let (sink, source) = linux::video_decoder_split(config, report_frame_decoded).unwrap();

        (
            VideoDecoderSink { inner: sink },
            VideoDecoderSource { inner: source },
        )
    }
    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    (VideoDecoderSink {}, VideoDecoderSource {})
}
