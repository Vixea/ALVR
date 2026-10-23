use super::{GraphicsContext, MAX_PUSH_CONSTANTS_SIZE, ck, staging::StagingRenderer};
use alvr_common::{
    AlvrFoveatedEncodingParams, ViewParams,
    glam::{Mat4, UVec2, Vec2, Vec3, Vec4},
};
use alvr_session::{PassthroughMode, UpscalingConfig};
use glow::{self as gl, HasContext};
use std::{cell::Cell, collections::VecDeque, ffi::c_void, iter, mem, rc::Rc, time::Duration};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, Color, ColorTargetState, ColorWrites, FragmentState, LoadOp,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology,
    PushConstantRange, RenderPass, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, StoreOp,
    Texture, TextureFormat, TextureSampleType, TextureView, TextureViewDescriptor,
    TextureViewDimension, VertexState, include_wgsl,
};

const FLOAT_SIZE: u32 = mem::size_of::<f32>() as u32;
const U32_SIZE: u32 = mem::size_of::<u32>() as u32;
const VEC4_SIZE: u32 = mem::size_of::<Vec4>() as u32;
const TRANSFORM_SIZE: u32 = mem::size_of::<Mat4>() as u32;

const TRANSFORM_CONST_OFFSET: u32 = 0;
const VIEW_INDEX_CONST_OFFSET: u32 = TRANSFORM_SIZE;
const PASSTHROUGH_MODE_OFFSET: u32 = VIEW_INDEX_CONST_OFFSET + U32_SIZE;
const ALPHA_CONST_OFFSET: u32 = PASSTHROUGH_MODE_OFFSET + U32_SIZE;
const CK_CHANNEL0_CONST_OFFSET: u32 = ALPHA_CONST_OFFSET + FLOAT_SIZE + U32_SIZE;
const CK_CHANNEL1_CONST_OFFSET: u32 = CK_CHANNEL0_CONST_OFFSET + VEC4_SIZE;
const CK_CHANNEL2_CONST_OFFSET: u32 = CK_CHANNEL1_CONST_OFFSET + VEC4_SIZE;
const PUSH_CONSTANTS_SIZE: u32 = CK_CHANNEL2_CONST_OFFSET + VEC4_SIZE;
const FOVEATION_EYE_UNIFORM_VEC4_COUNT: usize = 5;
const FOVEATION_UNIFORM_VEC4_COUNT: usize = 1 + FOVEATION_EYE_UNIFORM_VEC4_COUNT * 2;
const FOVEATION_UNIFORM_SIZE: usize = FOVEATION_UNIFORM_VEC4_COUNT * mem::size_of::<Vec4>();
const _: () = assert!(
    PUSH_CONSTANTS_SIZE <= MAX_PUSH_CONSTANTS_SIZE,
    "Push constants size exceeds the maximum size"
);

pub struct StreamViewParams {
    pub swapchain_index: u32,
    pub input_view_params: ViewParams,
    pub output_view_params: ViewParams,
}

pub struct StagingSlot {
    pub textures: [Texture; 2],
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub textures_gl: [gl::Texture; 2],
    pub framebuffers: [gl::Framebuffer; 2],
    pub bind_groups: [BindGroup; 2],
}

#[derive(Clone, Copy, Debug)]
pub struct QueuedFrame {
    pub slot_idx: usize,
    pub timestamp: Duration,
}

pub struct StreamRenderer {
    context: Rc<GraphicsContext>,
    staging_renderer: StagingRenderer,
    pipeline: RenderPipeline,
    render_targets: [Vec<TextureView>; 2],
    foveated_encoding: Option<AlvrFoveatedEncodingParams>,
    foveation_buffer: Buffer,
    last_foveation_center_shifts: Cell<Option<[Vec2; 2]>>,

    // Growable staging swapchain
    staging_resolution: UVec2,
    target_format: TextureFormat,
    bind_group_layout: BindGroupLayout,
    sampler: Sampler,
    staging_slots: Vec<StagingSlot>,
    free_slots: Vec<usize>,
    max_staging_slots: usize,
    ready_frames: VecDeque<QueuedFrame>,
    current_slot: Option<usize>,

    // Buffering control
    max_buffering_frames: f32,
    buffering_history_weight: f32,
    buffering_running_average: f32,
}

impl StreamRenderer {
    #[expect(clippy::too_many_arguments)]
    #[cfg_attr(any(target_os = "macos", target_os = "ios"), expect(unused))]
    pub fn new(
        context: Rc<GraphicsContext>,
        base_view_resolution: UVec2,
        target_view_resolution: UVec2,
        swapchain_textures: [Vec<u32>; 2],
        target_format: u32,
        foveated_encoding: Option<AlvrFoveatedEncodingParams>,
        enable_srgb_correction: bool,
        fix_limited_range: bool,
        encoding_gamma: f32,
        upscaling: Option<UpscalingConfig>,
        max_buffering_frames: f32,
        buffering_history_weight: f32,
    ) -> Self {
        let device = &context.device;

        let target_format = super::gl_format_to_wgpu(target_format);

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader_module = device.create_shader_module(include_wgsl!("../resources/stream.wgsl"));

        let mut constants = vec![];

        constants.extend([
            ("ENABLE_SRGB_CORRECTION", enable_srgb_correction.into()),
            ("ENCODING_GAMMA", encoding_gamma.into()),
        ]);

        let (staging_resolution, initial_foveation_uniforms) =
            if let Some(config) = &foveated_encoding {
                (
                    UVec2::from_array(config.encoded_view_resolution),
                    foveated_encoding_data(config, config.center_shifts.map(Vec2::from_array)),
                )
            } else {
                (
                    base_view_resolution,
                    [Vec4::ZERO; FOVEATION_UNIFORM_VEC4_COUNT],
                )
            };

        constants.push(("ENABLE_FFE", foveated_encoding.is_some().into()));

        let foveation_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("stream foveation uniforms"),
            size: FOVEATION_UNIFORM_SIZE as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context.queue.write_buffer(
            &foveation_buffer,
            0,
            &foveation_uniform_bytes(&initial_foveation_uniforms),
        );

        if foveated_encoding.is_none() {
            debug_assert_eq!(staging_resolution, base_view_resolution);
        }

        if let Some(upscaling) = upscaling {
            constants.extend([
                ("ENABLE_UPSCALING", true.into()),
                (
                    "UPSCALE_USE_EDGE_DIRECTION",
                    upscaling.edge_direction.into(),
                ),
                (
                    "UPSCALE_EDGE_THRESHOLD",
                    (upscaling.edge_threshold / 255.0).into(),
                ),
                ("UPSCALE_EDGE_SHARPNESS", upscaling.edge_sharpness.into()),
            ]);
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            // Note: Layout cannot be inferred because of a bug with push constants
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::VERTEX_FRAGMENT,
                    range: 0..PUSH_CONSTANTS_SIZE,
                }],
            })),
            vertex: VertexState {
                module: &shader_module,
                entry_point: None,
                compilation_options: PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: false,
                },
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: None,
                compilation_options: PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: false,
                },
                targets: &[Some(ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut render_targets = vec![];
        for target_swapchain in &swapchain_textures {
            let render_target = super::create_gl_swapchain(
                device,
                target_swapchain,
                target_view_resolution,
                target_format,
            );
            render_targets.push(render_target);
        }

        let staging_renderer = StagingRenderer::new(
            Rc::clone(&context),
            staging_resolution,
            fix_limited_range,
        );

        // Growable swapchain starts with 2 slots (steady-state ping-pong without wasting VRAM).
        // It grows on-demand when bursts or jitter deliver multiple frames before vsync,
        // allowing hardware decoder buffers (DMA-BUFs / AHardwareBuffers) to be ingested and freed
        // immediately without dropping frames or stalling the decoder.
        let initial_slots = 2;
        let max_staging_slots = (max_buffering_frames.ceil() as usize * 2 + 2).clamp(4, 16);
        let mut staging_slots = Vec::with_capacity(max_staging_slots);
        for _ in 0..initial_slots {
            staging_slots.push(Self::create_staging_slot(
                &staging_renderer,
                device,
                &bind_group_layout,
                &sampler,
                &foveation_buffer,
                staging_resolution,
                target_format,
            ));
        }
        let free_slots = (0..initial_slots).collect();

        Self {
            context,
            staging_renderer,
            pipeline,
            render_targets: render_targets.try_into().unwrap(),
            foveated_encoding,
            foveation_buffer,
            last_foveation_center_shifts: Cell::new(None),
            staging_resolution,
            target_format,
            bind_group_layout,
            sampler,
            staging_slots,
            free_slots,
            max_staging_slots,
            ready_frames: VecDeque::new(),
            current_slot: None,
            max_buffering_frames,
            buffering_history_weight,
            buffering_running_average: 0.0,
        }
    }

    fn create_staging_slot(
        staging_renderer: &StagingRenderer,
        device: &wgpu::Device,
        bind_group_layout: &BindGroupLayout,
        sampler: &Sampler,
        foveation_buffer: &Buffer,
        staging_resolution: UVec2,
        target_format: TextureFormat,
    ) -> StagingSlot {
        let mut textures = vec![];
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let mut textures_gl = vec![];
        let mut bind_groups = vec![];

        for _ in 0..2 {
            let staging_texture = super::create_texture(device, staging_resolution, target_format);

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(
                            &staging_texture.create_view(&TextureViewDescriptor::default()),
                        ),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: foveation_buffer.as_entire_binding(),
                    },
                ],
            });

            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            let staging_texture_gl = unsafe {
                staging_texture.as_hal::<wgpu::hal::api::Gles, _, _>(|tex| {
                    let wgpu::hal::gles::TextureInner::Texture { raw, .. } = tex.unwrap().inner
                    else {
                        panic!("invalid texture type");
                    };
                    raw
                })
            };

            textures.push(staging_texture);
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            textures_gl.push(staging_texture_gl);
            bind_groups.push(bind_group);
        }

        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let textures_gl: [gl::Texture; 2] = textures_gl.try_into().unwrap();
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let framebuffers = staging_renderer.create_framebuffers(&textures_gl);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let framebuffers = [glow::NativeFramebuffer(0), glow::NativeFramebuffer(0)];

        StagingSlot {
            textures: textures.try_into().unwrap(),
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            textures_gl,
            framebuffers,
            bind_groups: bind_groups.try_into().unwrap(),
        }
    }

    /// Ingests a decoded hardware frame into a staging texture in the growable swapchain,
    /// enabling the caller to immediately release the hardware decoder buffer.
    pub fn ingest_frame(&mut self, timestamp: Duration, hardware_buffer: *mut c_void) {
        if hardware_buffer.is_null() {
            return;
        }

        let slot_idx = if let Some(idx) = self.free_slots.pop() {
            idx
        } else if self.staging_slots.len() < self.max_staging_slots {
            let new_slot = Self::create_staging_slot(
                &self.staging_renderer,
                &self.context.device,
                &self.bind_group_layout,
                &self.sampler,
                &self.foveation_buffer,
                self.staging_resolution,
                self.target_format,
            );
            let idx = self.staging_slots.len();
            self.staging_slots.push(new_slot);
            alvr_common::info!(
                "Growing graphics staging swapchain to {} slots",
                self.staging_slots.len()
            );
            idx
        } else {
            // Swapchain is at max capacity: drop oldest buffered frame
            if let Some(oldest) = self.ready_frames.pop_front() {
                alvr_common::warn!(
                    "Graphics staging swapchain full ({}), dropping oldest buffered frame at {:?}",
                    self.max_staging_slots,
                    oldest.timestamp
                );
                oldest.slot_idx
            } else if let Some(curr) = self.current_slot {
                curr
            } else {
                return;
            }
        };

        // Render hardware buffer into the acquired slot's framebuffers
        self.staging_renderer
            .render(&self.staging_slots[slot_idx].framebuffers, hardware_buffer);

        self.ready_frames.push_back(QueuedFrame {
            slot_idx,
            timestamp,
        });
    }

    /// Selects the next frame to present according to the buffering policy, retiring the previously
    /// presented slot back to the free pool. Returns the presentation timestamp if a new frame is ready.
    pub fn acquire_frame_for_presentation(&mut self) -> Option<Duration> {
        if self.ready_frames.is_empty() {
            return None;
        }

        // Update running average of queue length
        self.buffering_running_average = self.buffering_running_average
            * self.buffering_history_weight
            + self.ready_frames.len() as f32 * (1. - self.buffering_history_weight);

        // Drop frames if running average exceeds max_buffering_frames
        while self.buffering_running_average > self.max_buffering_frames
            && self.ready_frames.len() > 1
        {
            if let Some(dropped) = self.ready_frames.pop_front() {
                self.free_slots.push(dropped.slot_idx);
            }
        }

        let frame = self.ready_frames.pop_front()?;
        if let Some(prev_idx) = self.current_slot.replace(frame.slot_idx) {
            if prev_idx != frame.slot_idx {
                self.free_slots.push(prev_idx);
            }
        }

        Some(frame.timestamp)
    }

    pub fn is_queue_empty(&self) -> bool {
        self.ready_frames.is_empty()
    }

    pub fn queued_frame_count(&self) -> usize {
        self.ready_frames.len()
    }

    pub fn has_current_frame(&self) -> bool {
        self.current_slot.is_some()
    }

    pub fn clear_queue(&mut self) {
        for frame in self.ready_frames.drain(..) {
            self.free_slots.push(frame.slot_idx);
        }
    }

    /// Runtime foveation centers must already be encoder-aligned. `None` uses the negotiated centers.
    ///
    /// # Safety
    /// If non-null, `hardware_buffer` must be a valid pointer to a ANativeWindowBuffer (Android)
    /// or LinuxVideoFrame (Linux).
    pub fn render(
        &mut self,
        hardware_buffer: *mut c_void,
        view_params: [StreamViewParams; 2],
        passthrough: Option<&PassthroughMode>,
        foveation_center_shifts: Option<[Vec2; 2]>,
    ) {
        // If hardware_buffer is passed directly, ingest and acquire it
        if !hardware_buffer.is_null() {
            self.ingest_frame(Duration::ZERO, hardware_buffer);
            self.acquire_frame_for_presentation();
        }

        let Some(slot_idx) = self.current_slot else {
            return;
        };

        if let Some(config) = &self.foveated_encoding
            && self.last_foveation_center_shifts.get() != foveation_center_shifts
        {
            let center_shifts = foveation_center_shifts
                .unwrap_or_else(|| config.center_shifts.map(Vec2::from_array));
            let uniforms = foveated_encoding_data(config, center_shifts);
            self.context.queue.write_buffer(
                &self.foveation_buffer,
                0,
                &foveation_uniform_bytes(&uniforms),
            );
            self.last_foveation_center_shifts
                .set(foveation_center_shifts);
        }

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&Default::default());

        for (view_idx, view_params) in view_params.iter().enumerate() {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.render_targets[view_idx]
                        [view_params.swapchain_index as usize],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            let input_fov = view_params.input_view_params.fov;

            let tanl = f32::tan(input_fov.left);
            let tanr = f32::tan(input_fov.right);
            let tanu = f32::tan(input_fov.up);
            let tand = f32::tan(input_fov.down);

            let width = tanr - tanl;
            let height = tanu - tand;
            let quad_depth = 1000.0;

            let output_mat4 = Mat4::from_translation(view_params.output_view_params.pose.position)
                * Mat4::from_quat(view_params.output_view_params.pose.orientation);
            let input_mat4 = Mat4::from_translation(view_params.input_view_params.pose.position)
                * Mat4::from_quat(view_params.input_view_params.pose.orientation);

            // The image is at z = -1.0, so we use tangents for the size
            let model_mat =
                Mat4::from_translation(Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: -quad_depth * 0.5,
                }) * Mat4::from_scale(Vec3::new(quad_depth, quad_depth, quad_depth * 0.5))
                    * Mat4::from_translation(Vec3::new(
                        width / 2.0 + tanl,
                        height / 2.0 + tand,
                        -1.0,
                    ))
                    * Mat4::from_scale(Vec3::new(width, height, 1.));
            let view_mat = output_mat4.inverse() * input_mat4;
            let proj_mat = super::projection_from_fov(view_params.output_view_params.fov);

            let transform = proj_mat * view_mat * model_mat;

            let transform_bytes = transform
                .to_cols_array()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>();

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_push_constants(
                ShaderStages::VERTEX_FRAGMENT,
                TRANSFORM_CONST_OFFSET,
                &transform_bytes,
            );
            render_pass.set_push_constants(
                ShaderStages::VERTEX_FRAGMENT,
                VIEW_INDEX_CONST_OFFSET,
                &(view_idx as u32).to_le_bytes(),
            );
            render_pass.set_bind_group(0, &self.staging_slots[slot_idx].bind_groups[view_idx], &[]);
            set_passthrough_push_constants(&mut render_pass, passthrough);
            render_pass.draw(0..4, 0..1);
        }

        self.context.queue.submit(iter::once(encoder.finish()));
    }
}

impl Drop for StreamRenderer {
    fn drop(&mut self) {
        let gl = &self.context.gl_context;
        self.context.make_current();

        unsafe {
            for slot in &self.staging_slots {
                for fb in slot.framebuffers {
                    ck!(gl.delete_framebuffer(fb));
                }
            }
        }
    }
}

fn set_passthrough_push_constants(render_pass: &mut RenderPass, config: Option<&PassthroughMode>) {
    const DEG_TO_NORM: f32 = 1. / 360.;

    fn set_u32(render_pass: &mut RenderPass, offset: u32, value: u32) {
        render_pass.set_push_constants(ShaderStages::VERTEX_FRAGMENT, offset, &value.to_le_bytes());
    }

    fn set_float(render_pass: &mut RenderPass, offset: u32, value: f32) {
        render_pass.set_push_constants(ShaderStages::VERTEX_FRAGMENT, offset, &value.to_le_bytes());
    }

    fn set_vec4(render_pass: &mut RenderPass, offset: u32, value: Vec4) {
        render_pass.set_push_constants(
            ShaderStages::VERTEX_FRAGMENT,
            offset,
            &value.x.to_le_bytes(),
        );
        render_pass.set_push_constants(
            ShaderStages::VERTEX_FRAGMENT,
            offset + FLOAT_SIZE,
            &value.y.to_le_bytes(),
        );
        render_pass.set_push_constants(
            ShaderStages::VERTEX_FRAGMENT,
            offset + 2 * FLOAT_SIZE,
            &value.z.to_le_bytes(),
        );
        render_pass.set_push_constants(
            ShaderStages::VERTEX_FRAGMENT,
            offset + 3 * FLOAT_SIZE,
            &value.w.to_le_bytes(),
        );
    }

    match config {
        None => {
            set_u32(render_pass, PASSTHROUGH_MODE_OFFSET, 0);
            set_float(render_pass, ALPHA_CONST_OFFSET, 1.);
        }
        Some(PassthroughMode::Blend { threshold, .. }) => {
            set_u32(render_pass, PASSTHROUGH_MODE_OFFSET, 0);
            set_float(render_pass, ALPHA_CONST_OFFSET, 1. - threshold);
        }
        Some(PassthroughMode::RgbChromaKey(config)) => {
            set_u32(render_pass, PASSTHROUGH_MODE_OFFSET, 1);

            let norm = |v| v as f32 / 255.;

            let red = norm(config.red);
            let green = norm(config.green);
            let blue = norm(config.blue);

            let thresh = norm(config.distance_threshold);

            let up_feather = 1. + config.feathering;
            let down_feather = 1. - config.feathering;

            let range_vec =
                thresh * Vec4::new(-up_feather, -down_feather, down_feather, up_feather);

            set_vec4(render_pass, CK_CHANNEL0_CONST_OFFSET, red + range_vec);
            set_vec4(render_pass, CK_CHANNEL1_CONST_OFFSET, green + range_vec);
            set_vec4(render_pass, CK_CHANNEL2_CONST_OFFSET, blue + range_vec);
        }
        Some(PassthroughMode::HsvChromaKey(config)) => {
            set_u32(render_pass, PASSTHROUGH_MODE_OFFSET, 2);

            set_vec4(
                render_pass,
                CK_CHANNEL0_CONST_OFFSET,
                Vec4::new(
                    config.hue_start_max_deg,
                    config.hue_start_min_deg,
                    config.hue_end_min_deg,
                    config.hue_end_max_deg,
                ) * DEG_TO_NORM,
            );

            set_vec4(
                render_pass,
                CK_CHANNEL1_CONST_OFFSET,
                Vec4::new(
                    config.saturation_start_max,
                    config.saturation_start_min,
                    config.saturation_end_min,
                    config.saturation_end_max,
                ),
            );

            set_vec4(
                render_pass,
                CK_CHANNEL2_CONST_OFFSET,
                Vec4::new(
                    config.value_start_max,
                    config.value_start_min,
                    config.value_end_min,
                    config.value_end_max,
                ),
            );
        }
    }
}

fn foveated_encoding_data(
    config: &AlvrFoveatedEncodingParams,
    center_shifts: [Vec2; 2],
) -> [Vec4; FOVEATION_UNIFORM_VEC4_COUNT] {
    let center_size_aligned = Vec2::from_array(config.center_size);
    let edge_ratio = Vec2::from_array(config.edge_ratio);
    let view_ratio_aligned = Vec2::from_array(config.view_ratio);

    let c0 = (1. - center_size_aligned) * 0.5;
    let c2 = (edge_ratio - 1.) * center_size_aligned + 1.;

    let mut uniforms = [Vec4::ZERO; FOVEATION_UNIFORM_VEC4_COUNT];
    let [common_uniform, eye_uniforms @ ..] = &mut uniforms;
    *common_uniform = Vec4::new(
        view_ratio_aligned.x,
        view_ratio_aligned.y,
        edge_ratio.x,
        edge_ratio.y,
    );

    for (uniforms, center_shift) in eye_uniforms
        .chunks_exact_mut(FOVEATION_EYE_UNIFORM_VEC4_COUNT)
        .zip(center_shifts)
    {
        let c1 = (edge_ratio - 1.) * c0 * (center_shift + 1.) / edge_ratio;
        let lo_bound = c0 * (center_shift + 1.);
        let hi_bound = c0 * (center_shift - 1.) + 1.;
        let lo_bound_c = c0 * (center_shift + 1.) / c2;
        let hi_bound_c = c0 * (center_shift - 1.) / c2 + 1.;

        let a_left = c2 * (1. - edge_ratio) / (edge_ratio * lo_bound_c);
        let b_left = (c1 + c2 * lo_bound_c) / lo_bound_c;

        let a_right = c2 * (edge_ratio - 1.) / (edge_ratio * (1. - hi_bound_c));
        let b_right = (c2 - edge_ratio * c1 - 2. * edge_ratio * c2
            + c2 * edge_ratio * (1. - hi_bound_c)
            + edge_ratio)
            / (edge_ratio * (1. - hi_bound_c));
        let c_right = (c2 * edge_ratio - c2) * (c1 - hi_bound_c + c2 * hi_bound_c)
            / (edge_ratio * (1. - hi_bound_c) * (1. - hi_bound_c));

        let [center, bounds, left, right, right_curve] = uniforms else {
            unreachable!("foveation eye uniform chunk must contain five vectors");
        };
        *center = Vec4::new(c1.x, c1.y, c2.x, c2.y);
        *bounds = Vec4::new(lo_bound.x, lo_bound.y, hi_bound.x, hi_bound.y);
        *left = Vec4::new(a_left.x, a_left.y, b_left.x, b_left.y);
        *right = Vec4::new(a_right.x, a_right.y, b_right.x, b_right.y);
        *right_curve = Vec4::new(c_right.x, c_right.y, 0.0, 0.0);
    }

    uniforms
}

fn foveation_uniform_bytes(
    uniforms: &[Vec4; FOVEATION_UNIFORM_VEC4_COUNT],
) -> [u8; FOVEATION_UNIFORM_SIZE] {
    let mut bytes = [0; FOVEATION_UNIFORM_SIZE];

    for (chunk, value) in bytes
        .chunks_exact_mut(mem::size_of::<f32>())
        .zip(uniforms.iter().flat_map(|value| value.to_array()))
    {
        chunk.copy_from_slice(&value.to_ne_bytes());
    }

    bytes
}

pub fn compute_target_view_resolution(
    resolution: UVec2,
    upscaling: &Option<UpscalingConfig>,
) -> UVec2 {
    let mut target_resolution = resolution.as_vec2();
    if let Some(upscaling) = upscaling {
        target_resolution *= upscaling.upscale_factor;
    }
    target_resolution.as_uvec2()
}
