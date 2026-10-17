// Linux video decoder based on FFmpeg.
//
// Hardware decoding uses the V4L2 memory-to-memory decoders (h264_v4l2m2m / hevc_v4l2m2m), which
// on Snapdragon devices like the Steam Frame are backed by the Qualcomm Venus/Iris stateful decoder.
// When possible frames are exported as DRM PRIME (DMA-BUF) and imported zero-copy with EGL. Any
// other output (software decoders, non DRM PRIME formats) is converted to RGBA with swscale.

use super::VideoDecoderConfig;
use alvr_common::{
    RelaxedAtomic,
    anyhow::{Result, anyhow, bail},
    error, info,
    parking_lot::Mutex,
    warn,
};
use alvr_graphics::{DMA_BUF_MAX_PLANES, DmaBufImage, DmaBufPlane, LinuxVideoFrame};
use alvr_session::CodecType;
use ffmpeg_next::{
    self as ffmpeg, Packet, codec, ffi,
    format::Pixel,
    frame,
    software::scaling::{self, Flags as ScalingFlags},
    util::error::EAGAIN,
};
use std::{
    collections::VecDeque,
    ffi::{c_int, c_void},
    ptr,
    sync::{
        Arc,
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

const INPUT_QUEUE_SIZE: usize = 8;
const INPUT_RECV_TIMEOUT: Duration = Duration::from_millis(10);
const SEND_RETRY_INTERVAL: Duration = Duration::from_micros(500);
const MAX_SEND_RETRIES: usize = 200; // 100ms
const TIMESTAMP_HISTORY_SIZE: usize = 128;
const MAX_CONSECUTIVE_ERRORS: usize = 10;

// FFmpeg pts are an incrementing packet index in microsecond units. V4L2 only carries timestamps
// with microsecond precision, so the real nanosecond timestamps are kept on our side.
const PTS_TIME_BASE: ffi::AVRational = ffi::AVRational {
    num: 1,
    den: 1_000_000,
};

const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
const fn fourcc(code: &[u8; 4]) -> u32 {
    (code[0] as u32) | (code[1] as u32) << 8 | (code[2] as u32) << 16 | (code[3] as u32) << 24
}
const DRM_FORMAT_NV12: u32 = fourcc(b"NV12");
const DRM_FORMAT_P010: u32 = fourcc(b"P010");
const DRM_FORMAT_R8: u32 = fourcc(b"R8  ");
const DRM_FORMAT_GR88: u32 = fourcc(b"GR88");
const DRM_FORMAT_R16: u32 = fourcc(b"R16 ");
const DRM_FORMAT_GR1616: u32 = fourcc(b"GR32");

// Mirrors of libavutil/hwcontext_drm.h, which is a stable public ABI. Defined here because
// ffmpeg-sys-next does not always generate bindings for it.
const AV_DRM_MAX_PLANES: usize = 4;

#[repr(C)]
struct AVDRMObjectDescriptor {
    fd: c_int,
    size: usize,
    format_modifier: u64,
}

#[repr(C)]
struct AVDRMPlaneDescriptor {
    object_index: c_int,
    offset: isize,
    pitch: isize,
}

#[repr(C)]
struct AVDRMLayerDescriptor {
    format: u32,
    nb_planes: c_int,
    planes: [AVDRMPlaneDescriptor; AV_DRM_MAX_PLANES],
}

#[repr(C)]
struct AVDRMFrameDescriptor {
    nb_objects: c_int,
    objects: [AVDRMObjectDescriptor; AV_DRM_MAX_PLANES],
    nb_layers: c_int,
    layers: [AVDRMLayerDescriptor; AV_DRM_MAX_PLANES],
}

pub struct DecodedFrame {
    timestamp: Duration,
    image: LinuxVideoFrame,
    // Keeps the DMA-BUFs alive (and out of the decoder pool) while the image is in use
    _av_frame: Option<frame::Video>,
}

pub struct VideoDecoderSink {
    packet_sender: SyncSender<(Duration, Vec<u8>)>,
    config_nal: Option<Vec<u8>>,
}

impl VideoDecoderSink {
    // Returns false if the decoder input queue is full
    pub fn push_frame_nal(&mut self, timestamp: Duration, data: &[u8]) -> Result<bool> {
        let buffer = if let Some(config_nal) = self.config_nal.take() {
            // Send parameter sets in-band with the first frame. Some V4L2 decoders ignore
            // extradata.
            [config_nal.as_slice(), data].concat()
        } else {
            data.to_vec()
        };

        match self.packet_sender.try_send((timestamp, buffer)) {
            Ok(()) => Ok(true),
            Err(TrySendError::Full(_)) => Ok(false),
            Err(TrySendError::Disconnected(_)) => bail!("Decoder thread stopped"),
        }
    }
}

pub struct VideoDecoderSource {
    running: Arc<RelaxedAtomic>,
    decoder_thread: Option<JoinHandle<()>>,
    frame_queue: Arc<Mutex<VecDeque<DecodedFrame>>>,
    current_frame: Option<DecodedFrame>,
    config: VideoDecoderConfig,
    buffering_running_average: f32,
}

impl VideoDecoderSource {
    // The returned pointer (to a LinuxVideoFrame) is valid until the next call of this function
    pub fn dequeue_frame(&mut self) -> Option<(Duration, *mut c_void)> {
        let mut frame_queue_lock = self.frame_queue.lock();

        // use running average to give more weight to recent samples
        self.buffering_running_average = self.buffering_running_average
            * self.config.buffering_history_weight
            + frame_queue_lock.len() as f32 * (1. - self.config.buffering_history_weight);
        if self.buffering_running_average > self.config.max_buffering_frames {
            frame_queue_lock.pop_front();
        }

        // Release the previous frame now that the renderer is done with it
        let frame = frame_queue_lock.pop_front();
        drop(frame_queue_lock);
        self.current_frame = None;

        let frame = self.current_frame.insert(frame?);

        Some((
            frame.timestamp,
            ptr::from_ref(&frame.image).cast_mut().cast(),
        ))
    }
}

impl Drop for VideoDecoderSource {
    fn drop(&mut self) {
        self.running.set(false);
        self.current_frame = None;

        if let Some(thread) = self.decoder_thread.take() {
            thread.join().ok();
        }
    }
}

unsafe extern "C" fn get_format(
    _ctx: *mut ffi::AVCodecContext,
    formats: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    let mut available = vec![];
    let mut format_ptr = formats;
    unsafe {
        while *format_ptr != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            available.push(*format_ptr);
            format_ptr = format_ptr.add(1);
        }
    }

    // Prefer zero-copy DRM PRIME, then the first (native) software format
    if available.contains(&ffi::AVPixelFormat::AV_PIX_FMT_DRM_PRIME) {
        ffi::AVPixelFormat::AV_PIX_FMT_DRM_PRIME
    } else {
        available
            .first()
            .copied()
            .unwrap_or(ffi::AVPixelFormat::AV_PIX_FMT_NONE)
    }
}

fn open_decoder(
    codec: ffmpeg::Codec,
    config_nal: &[u8],
) -> Result<ffmpeg::decoder::Video, ffmpeg::Error> {
    let mut context = codec::Context::new_with_codec(codec);
    context.set_flags(codec::Flags::LOW_DELAY);
    context.set_time_base(PTS_TIME_BASE);

    unsafe {
        let ctx = context.as_mut_ptr();

        (*ctx).pkt_timebase = PTS_TIME_BASE;
        (*ctx).get_format = Some(get_format);
        (*ctx).flags2 |= ffi::AV_CODEC_FLAG2_FAST as c_int;

        if !config_nal.is_empty() {
            let padding = ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize;
            let extradata = ffi::av_mallocz(config_nal.len() + padding).cast::<u8>();
            if !extradata.is_null() {
                ptr::copy_nonoverlapping(config_nal.as_ptr(), extradata, config_nal.len());
                (*ctx).extradata = extradata;
                (*ctx).extradata_size = config_nal.len() as c_int;
            }
        }
    }

    context.decoder().video()
}

fn create_decoder(config: &VideoDecoderConfig) -> Result<ffmpeg::decoder::Video> {
    let (hw_name, sw_id) = match config.codec {
        CodecType::H264 => ("h264_v4l2m2m", codec::Id::H264),
        CodecType::Hevc => ("hevc_v4l2m2m", codec::Id::HEVC),
        CodecType::AV1 => bail!("AV1 decoding is not supported on Linux yet"),
    };

    if !config.force_software_decoder {
        match ffmpeg::decoder::find_by_name(hw_name) {
            Some(codec) => match open_decoder(codec, &config.config_buffer) {
                Ok(decoder) => {
                    info!("Using hardware decoder {hw_name}");
                    return Ok(decoder);
                }
                // would be "warn!" but this is a severe caveat and a pretty major error.
                Err(e) => error!("Attempting software fallback due to error in {hw_name}: {e}"),
            },
            None => error!("Hardware decoder {hw_name} not available, using software decoding"),
        }
    }

    let codec = ffmpeg::decoder::find(sw_id)
        .ok_or_else(|| anyhow!("Software decoder for {sw_id:?} not available"))?;
    let decoder = open_decoder(codec, &config.config_buffer)?;
    info!("Using software decoder {}", codec.name());

    Ok(decoder)
}

fn dma_buf_image(frame: &frame::Video) -> Option<DmaBufImage> {
    let descriptor = unsafe {
        let data = (*frame.as_ptr()).data[0];
        if data.is_null() {
            return None;
        }
        &*data.cast::<AVDRMFrameDescriptor>()
    };

    let object_count = (descriptor.nb_objects as usize).min(AV_DRM_MAX_PLANES);
    let layer_count = (descriptor.nb_layers as usize).min(AV_DRM_MAX_PLANES);
    if layer_count == 0 {
        return None;
    }

    let mut image = DmaBufImage {
        width: frame.width(),
        height: frame.height(),
        ..Default::default()
    };

    // Flatten all layers into a single multi-planar image
    for layer in &descriptor.layers[..layer_count] {
        for plane in &layer.planes[..(layer.nb_planes as usize).min(AV_DRM_MAX_PLANES)] {
            let object_index = plane.object_index as usize;
            if object_index >= object_count || image.plane_count >= DMA_BUF_MAX_PLANES {
                return None;
            }
            let object = &descriptor.objects[object_index];

            image.planes[image.plane_count] = DmaBufPlane {
                fd: object.fd,
                offset: plane.offset as u32,
                pitch: plane.pitch as u32,
                modifier: (object.format_modifier != DRM_FORMAT_MOD_INVALID)
                    .then_some(object.format_modifier),
            };
            image.plane_count += 1;
        }
    }

    // Separate luma/chroma layers are imported as a single biplanar YUV image
    image.fourcc = if layer_count == 1 {
        descriptor.layers[0].format
    } else {
        match (descriptor.layers[0].format, descriptor.layers[1].format) {
            (DRM_FORMAT_R8, DRM_FORMAT_GR88) => DRM_FORMAT_NV12,
            (DRM_FORMAT_R16, DRM_FORMAT_GR1616) => DRM_FORMAT_P010,
            (y, uv) => {
                warn!("Unsupported DRM PRIME layer formats: {y:#x}, {uv:#x}");
                return None;
            }
        }
    };

    Some(image)
}

struct RgbaConverter {
    scaler: Option<(scaling::Context, Pixel, u32, u32)>,
    rgba_frame: frame::Video,
}

impl RgbaConverter {
    fn new() -> Self {
        Self {
            scaler: None,
            rgba_frame: frame::Video::empty(),
        }
    }

    fn convert(&mut self, frame: &frame::Video) -> Result<LinuxVideoFrame> {
        let (format, width, height) = (frame.format(), frame.width(), frame.height());

        let needs_new_scaler = !matches!(
            &self.scaler,
            Some((_, f, w, h)) if *f == format && *w == width && *h == height
        );
        if needs_new_scaler {
            let mut scaler = scaling::Context::get(
                format,
                width,
                height,
                Pixel::RGBA,
                width,
                height,
                ScalingFlags::BILINEAR,
            )?;

            // Match the BT.709 limited range hint used for the DMA-BUF path
            unsafe {
                let coefficients = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709 as c_int);
                ffi::sws_setColorspaceDetails(
                    scaler.as_mut_ptr(),
                    coefficients,
                    0,
                    coefficients,
                    1,
                    0,
                    1 << 16,
                    1 << 16,
                );
            }

            self.scaler = Some((scaler, format, width, height));
            self.rgba_frame = frame::Video::new(Pixel::RGBA, width, height);
        }

        let (scaler, ..) = self.scaler.as_mut().unwrap();
        scaler.run(frame, &mut self.rgba_frame)?;

        // Repack tightly, the scaler output may have row padding
        let stride = self.rgba_frame.stride(0);
        let row_size = width as usize * 4;
        let src = self.rgba_frame.data(0);
        let mut data = Vec::with_capacity(row_size * height as usize);
        for row in 0..height as usize {
            data.extend_from_slice(&src[row * stride..row * stride + row_size]);
        }

        Ok(LinuxVideoFrame::Rgba {
            width,
            height,
            data,
        })
    }
}

struct DecoderLoop<F> {
    decoder: ffmpeg::decoder::Video,
    frame_queue: Arc<Mutex<VecDeque<DecodedFrame>>>,
    frame_result_callback: Arc<F>,
    timestamps: VecDeque<(i64, Duration)>,
    next_pts: i64,
    converter: RgbaConverter,
    max_queued_frames: usize,
    consecutive_errors: usize,
}

impl<F: Fn(Result<Duration>) + Send + Sync + 'static> DecoderLoop<F> {
    fn send_packet(&mut self, timestamp: Duration, data: &[u8]) -> Result<()> {
        let pts = self.next_pts;
        self.next_pts += 1;

        self.timestamps.push_back((pts, timestamp));
        while self.timestamps.len() > TIMESTAMP_HISTORY_SIZE {
            self.timestamps.pop_front();
        }

        let mut packet = Packet::copy(data);
        packet.set_pts(Some(pts));
        packet.set_dts(Some(pts));

        for _ in 0..MAX_SEND_RETRIES {
            match self.decoder.send_packet(&packet) {
                Ok(()) => {
                    self.consecutive_errors = 0;
                    return Ok(());
                }
                Err(ffmpeg::Error::Other { errno: EAGAIN }) => {
                    // Decoder input is full, drain output to make room
                    self.receive_frames();
                    thread::sleep(SEND_RETRY_INTERVAL);
                }
                Err(e) => {
                    self.consecutive_errors += 1;
                    if self.consecutive_errors > MAX_CONSECUTIVE_ERRORS {
                        bail!("Too many decoder errors: {e}");
                    }
                    warn!("Decoder error: {e}");

                    return Ok(());
                }
            }
        }

        warn!("Dropped video packet. Reason: decoder input stalled");

        Ok(())
    }

    fn receive_frames(&mut self) {
        loop {
            let mut av_frame = frame::Video::empty();
            match self.decoder.receive_frame(&mut av_frame) {
                Ok(()) => self.push_frame(av_frame),
                Err(ffmpeg::Error::Other { errno: EAGAIN }) | Err(ffmpeg::Error::Eof) => break,
                Err(e) => {
                    warn!("Decoder receive error: {e}");
                    break;
                }
            }
        }
    }

    fn push_frame(&mut self, av_frame: frame::Video) {
        let Some(pts) = av_frame.pts().or(av_frame.timestamp()) else {
            warn!("Decoded frame without timestamp");
            return;
        };
        let Some(timestamp) = self
            .timestamps
            .iter()
            .find_map(|(p, timestamp)| (*p == pts).then_some(*timestamp))
        else {
            warn!("Decoded frame with unknown timestamp");
            return;
        };

        let (image, av_frame) = if av_frame.format() == Pixel::DRM_PRIME {
            match dma_buf_image(&av_frame) {
                Some(image) => (LinuxVideoFrame::DmaBuf(image), Some(av_frame)),
                None => {
                    error!("Invalid DRM PRIME frame");
                    return;
                }
            }
        } else {
            match self.converter.convert(&av_frame) {
                Ok(image) => (image, None),
                Err(e) => {
                    error!("Frame conversion error: {e}");
                    return;
                }
            }
        };

        (self.frame_result_callback)(Ok(timestamp));

        let mut frame_queue_lock = self.frame_queue.lock();
        if frame_queue_lock.len() > self.max_queued_frames {
            warn!("Video frame queue overflow!");
            frame_queue_lock.pop_front();
        }
        frame_queue_lock.push_back(DecodedFrame {
            timestamp,
            image,
            _av_frame: av_frame,
        });
    }

    fn run(
        &mut self,
        running: &RelaxedAtomic,
        packet_receiver: &Receiver<(Duration, Vec<u8>)>,
    ) -> Result<()> {
        while running.value() {
            match packet_receiver.recv_timeout(INPUT_RECV_TIMEOUT) {
                Ok((timestamp, data)) => self.send_packet(timestamp, &data)?,
                Err(RecvTimeoutError::Timeout) => (),
                Err(RecvTimeoutError::Disconnected) => break,
            }

            self.receive_frames();
        }

        Ok(())
    }
}

// Create a sink/source pair
pub fn video_decoder_split(
    config: VideoDecoderConfig,
    frame_result_callback: impl Fn(Result<Duration>) + Send + Sync + 'static,
) -> Result<(VideoDecoderSink, VideoDecoderSource)> {
    ffmpeg::init()?;

    let running = Arc::new(RelaxedAtomic::new(true));
    let frame_queue = Arc::new(Mutex::new(VecDeque::new()));
    let (packet_sender, packet_receiver) = mpsc::sync_channel(INPUT_QUEUE_SIZE);
    let (ready_sender, ready_receiver) = mpsc::channel();

    // The decoder is created in its own thread since FFmpeg contexts are not thread safe
    let decoder_thread = thread::spawn({
        let config = config.clone();
        let running = Arc::clone(&running);
        let frame_queue = Arc::clone(&frame_queue);
        move || {
            let decoder = match create_decoder(&config) {
                Ok(decoder) => {
                    ready_sender.send(Ok(())).ok();
                    decoder
                }
                Err(e) => {
                    ready_sender.send(Err(e)).ok();
                    return;
                }
            };

            let frame_result_callback = Arc::new(frame_result_callback);

            let mut decoder_loop = DecoderLoop {
                decoder,
                frame_queue: Arc::clone(&frame_queue),
                frame_result_callback: Arc::clone(&frame_result_callback),
                timestamps: VecDeque::new(),
                next_pts: 0,
                converter: RgbaConverter::new(),
                // 2x: keep the target buffering in the middle of the max amount of queuable frames
                max_queued_frames: (2. * config.max_buffering_frames).ceil() as usize,
                consecutive_errors: 0,
            };

            if let Err(e) = decoder_loop.run(&running, &packet_receiver) {
                frame_result_callback(Err(e));
            }

            // Release all frames before the decoder is destroyed
            frame_queue.lock().clear();
        }
    });

    // Make sure the decoder is ready: we don't want to try to enqueue frame and lose them, to avoid
    // image corruption.
    ready_receiver
        .recv()
        .map_err(|_| anyhow!("Decoder thread exited unexpectedly"))??;

    let sink = VideoDecoderSink {
        packet_sender,
        config_nal: (!config.config_buffer.is_empty()).then(|| config.config_buffer.clone()),
    };
    let source = VideoDecoderSource {
        running,
        decoder_thread: Some(decoder_thread),
        frame_queue,
        current_frame: None,
        config,
        buffering_running_average: 0.0,
    };

    Ok((sink, source))
}
