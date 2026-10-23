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

const INPUT_QUEUE_SIZE: usize = 32;
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
}

impl VideoDecoderSource {
    // The returned pointer (to a LinuxVideoFrame) is valid until the next call of this function
    // or until release_frame() is called.
    pub fn dequeue_frame(&mut self) -> Option<(Duration, *mut c_void)> {
        self.current_frame = None;

        let mut frame_queue_lock = self.frame_queue.lock();
        let frame = frame_queue_lock.pop_front()?;
        drop(frame_queue_lock);

        let frame = self.current_frame.insert(frame);

        Some((
            frame.timestamp,
            ptr::from_ref(&frame.image).cast_mut().cast(),
        ))
    }

    pub fn release_frame(&mut self) {
        self.current_frame = None;
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
    ctx: *mut ffi::AVCodecContext,
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

    // Prefer zero-copy DRM PRIME when offered by v4l2m2m
    if available.contains(&ffi::AVPixelFormat::AV_PIX_FMT_DRM_PRIME) {
        return ffi::AVPixelFormat::AV_PIX_FMT_DRM_PRIME;
    }

    // Otherwise pick a native software pixel format and avoid probing unconfigured hwaccels
    // (vaapi, vdpau, cuda, vulkan)
    for preferred in [
        ffi::AVPixelFormat::AV_PIX_FMT_NV12,
        ffi::AVPixelFormat::AV_PIX_FMT_YUV420P,
        ffi::AVPixelFormat::AV_PIX_FMT_P010LE,
        ffi::AVPixelFormat::AV_PIX_FMT_YUV420P10LE,
    ] {
        if available.contains(&preferred) {
            return preferred;
        }
    }

    unsafe { ffi::avcodec_default_get_format(ctx, formats) }
}

fn create_drm_hwdevice_ctx() -> *mut ffi::AVBufferRef {
    unsafe {
        let mut hw_device_ctx: *mut ffi::AVBufferRef = ptr::null_mut();
        for path in [
            c"/dev/dri/renderD128".as_ptr(),
            c"/dev/dri/card0".as_ptr(),
            c"/dev/dri/card1".as_ptr(),
            ptr::null(),
        ] {
            if ffi::av_hwdevice_ctx_create(
                &mut hw_device_ctx,
                ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_DRM,
                path,
                ptr::null_mut(),
                0,
            ) >= 0
                && !hw_device_ctx.is_null()
            {
                return hw_device_ctx;
            }
        }

        // Fallback: v4l2_m2m only checks that hw_device_ctx->type == AV_HWDEVICE_TYPE_DRM to set
        // output_drm = 1 and export V4L2 capture buffers via VIDIOC_EXPBUF as AV_PIX_FMT_DRM_PRIME.
        hw_device_ctx = ffi::av_hwdevice_ctx_alloc(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_DRM);
        if !hw_device_ctx.is_null() && ffi::av_hwdevice_ctx_init(hw_device_ctx) < 0 {
            ffi::av_buffer_unref(&mut hw_device_ctx);
        }
        hw_device_ctx
    }
}

fn open_decoder(
    codec: ffmpeg::Codec,
    config_nal: &[u8],
    dimensions: Option<(c_int, c_int, c_int, c_int)>,
    is_hw: bool,
) -> Result<ffmpeg::decoder::Video, ffmpeg::Error> {
    let mut context = codec::Context::new_with_codec(codec);
    context.set_flags(codec::Flags::LOW_DELAY);
    context.set_time_base(PTS_TIME_BASE);

    unsafe {
        let ctx = context.as_mut_ptr();

        if let Some((width, height, coded_width, coded_height)) = dimensions {
            (*ctx).width = width;
            (*ctx).height = height;
            (*ctx).coded_width = coded_width;
            (*ctx).coded_height = coded_height;
        }

        (*ctx).pkt_timebase = PTS_TIME_BASE;
        (*ctx).get_format = Some(get_format);
        (*ctx).flags2 |= ffi::AV_CODEC_FLAG2_FAST as c_int;

        if is_hw {
            let hw_device_ctx = create_drm_hwdevice_ctx();
            if !hw_device_ctx.is_null() {
                (*ctx).hw_device_ctx = hw_device_ctx;
            }
        } else {
            (*ctx).thread_count = 4;
            (*ctx).thread_type = ffi::FF_THREAD_SLICE as c_int;
        }

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

// V4L2 M2M decoders require non-zero coded_width/coded_height on AVCodecContext before
// avcodec_open2 so VIDIOC_S_FMT and VIDIOC_REQBUFS on the output queue succeed (e.g. on Qualcomm
// iris_driver). Probe the SPS/VPS/PPS in config_nal using the software decoder first.
fn probe_dimensions(sw_id: codec::Id, config_nal: &[u8]) -> (c_int, c_int, c_int, c_int) {
    if let Some(codec) = ffmpeg::decoder::find(sw_id)
        && let Ok(mut decoder) = open_decoder(codec, config_nal, None, false)
    {
        if (decoder.width() == 0 || decoder.height() == 0) && !config_nal.is_empty() {
            let packet = Packet::copy(config_nal);
            let _ = decoder.send_packet(&packet);
        }

        let (width, height, coded_width, coded_height) = unsafe {
            let ctx = decoder.as_ptr();
            (
                (*ctx).width,
                (*ctx).height,
                (*ctx).coded_width,
                (*ctx).coded_height,
            )
        };

        if width > 0 && height > 0 {
            let cw = if coded_width > 0 { coded_width } else { width };
            let ch = if coded_height > 0 {
                coded_height
            } else {
                height
            };
            info!("Probed video dimensions from config NAL: {width}x{height} (coded: {cw}x{ch})");
            return (width, height, cw, ch);
        }
    }

    warn!("Could not probe dimensions from config NAL, defaulting to 1920x1080");
    (1920, 1080, 1920, 1080)
}

fn open_software_decoder(
    config: &VideoDecoderConfig,
    dimensions: (c_int, c_int, c_int, c_int),
) -> Result<ffmpeg::decoder::Video> {
    let sw_id = match config.codec {
        CodecType::H264 => codec::Id::H264,
        CodecType::Hevc => codec::Id::HEVC,
        CodecType::AV1 => bail!("AV1 decoding is not supported on Linux yet"),
    };

    let codec = ffmpeg::decoder::find(sw_id)
        .ok_or_else(|| anyhow!("Software decoder for {sw_id:?} not available"))?;
    let decoder = open_decoder(codec, &config.config_buffer, Some(dimensions), false)?;
    info!("Using software decoder {}", codec.name());

    Ok(decoder)
}

fn create_decoder(
    config: &VideoDecoderConfig,
) -> Result<(ffmpeg::decoder::Video, (c_int, c_int, c_int, c_int), bool)> {
    let (hw_name, sw_id) = match config.codec {
        CodecType::H264 => ("h264_v4l2m2m", codec::Id::H264),
        CodecType::Hevc => ("hevc_v4l2m2m", codec::Id::HEVC),
        CodecType::AV1 => bail!("AV1 decoding is not supported on Linux yet"),
    };

    let dimensions = probe_dimensions(sw_id, &config.config_buffer);

    if !config.force_software_decoder {
        match ffmpeg::decoder::find_by_name(hw_name) {
            Some(codec) => match open_decoder(codec, &config.config_buffer, Some(dimensions), true)
            {
                Ok(decoder) => {
                    info!("Using hardware decoder {hw_name}");
                    return Ok((decoder, dimensions, true));
                }
                // would be "warn!" but this is a severe caveat and a pretty major error.
                Err(e) => error!("Attempting software fallback due to error in {hw_name}: {e}"),
            },
            None => error!("Hardware decoder {hw_name} not available, using software decoding"),
        }
    }

    let decoder = open_software_decoder(config, dimensions)?;
    Ok((decoder, dimensions, false))
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

    // Single-plane contiguous V4L2_PIX_FMT_NV12 / P010 exported with 1 plane descriptor still needs
    // a second plane descriptor for EGL_EXT_image_dma_buf_import.
    if (image.fourcc == DRM_FORMAT_NV12 || image.fourcc == DRM_FORMAT_P010)
        && image.plane_count == 1
    {
        let plane0 = image.planes[0];
        image.planes[1] = DmaBufPlane {
            fd: plane0.fd,
            offset: plane0.offset + plane0.pitch * image.height,
            pitch: plane0.pitch,
            modifier: plane0.modifier,
        };
        image.plane_count = 2;
    }

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
    config: VideoDecoderConfig,
    dimensions: (c_int, c_int, c_int, c_int),
    is_hw: bool,
    decoded_any_frame: bool,
    frame_queue: Arc<Mutex<VecDeque<DecodedFrame>>>,
    frame_result_callback: Arc<F>,
    timestamps: VecDeque<(i64, Duration)>,
    next_pts: i64,
    converter: RgbaConverter,
    max_queued_frames: usize,
    consecutive_errors: usize,
}

impl<F: Fn(Result<Duration>) + Send + Sync + 'static> DecoderLoop<F> {
    fn fallback_to_software(&mut self, packet: &Packet) -> bool {
        if !self.is_hw {
            return false;
        }
        self.is_hw = false;
        match open_software_decoder(&self.config, self.dimensions) {
            Ok(sw_decoder) => {
                self.decoder = sw_decoder;
                self.consecutive_errors = 0;
                if !self.config.config_buffer.is_empty() {
                    let mut cfg_pkt = Packet::copy(&self.config.config_buffer);
                    cfg_pkt.set_pts(packet.pts());
                    cfg_pkt.set_dts(packet.dts());
                    let _ = self.decoder.send_packet(&cfg_pkt);
                }
                let _ = self.decoder.send_packet(packet);
                true
            }
            Err(e) => {
                error!("Failed to open software fallback decoder: {e}");
                false
            }
        }
    }

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
                    if self.is_hw && !self.decoded_any_frame {
                        error!("Hardware decoder send_packet failed ({e}), switching to software");
                        if self.fallback_to_software(&packet) {
                            return Ok(());
                        }
                    }
                    self.consecutive_errors += 1;
                    if self.consecutive_errors > MAX_CONSECUTIVE_ERRORS {
                        if self.fallback_to_software(&packet) {
                            return Ok(());
                        }
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
                Ok(()) => {
                    self.decoded_any_frame = true;
                    self.push_frame(av_frame);
                }
                Err(ffmpeg::Error::Other { errno: EAGAIN }) | Err(ffmpeg::Error::Eof) => break,
                Err(e) => {
                    warn!("Decoder receive error: {e}");
                    if self.is_hw && !self.decoded_any_frame {
                        self.consecutive_errors += 1;
                        if self.consecutive_errors >= 3 {
                            error!(
                                "Hardware decoder receive_frame failed repeatedly ({e}), switching to software"
                            );
                            let empty = Packet::empty();
                            self.fallback_to_software(&empty);
                        }
                    }
                    break;
                }
            }
        }
    }

    fn push_frame(&mut self, av_frame: frame::Video) {
        let timestamp = if let Some(pts) = av_frame.pts().or(av_frame.timestamp())
            && let Some(pos) = self.timestamps.iter().position(|(p, _)| *p == pts)
        {
            let (_, ts) = self.timestamps.remove(pos).unwrap();
            ts
        } else if let Some((_, ts)) = self.timestamps.pop_front() {
            ts
        } else {
            warn!("Decoded frame without matching timestamp");
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
            let (decoder, dimensions, is_hw) = match create_decoder(&config) {
                Ok(res) => {
                    ready_sender.send(Ok(())).ok();
                    res
                }
                Err(e) => {
                    ready_sender.send(Err(e)).ok();
                    return;
                }
            };

            let frame_result_callback = Arc::new(frame_result_callback);
            let max_queued_frames = 16;

            let mut decoder_loop = DecoderLoop {
                decoder,
                config,
                dimensions,
                is_hw,
                decoded_any_frame: false,
                frame_queue: Arc::clone(&frame_queue),
                frame_result_callback: Arc::clone(&frame_result_callback),
                timestamps: VecDeque::new(),
                next_pts: 1,
                converter: RgbaConverter::new(),
                max_queued_frames,
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
    };

    Ok((sink, source))
}
