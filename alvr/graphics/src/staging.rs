use super::{GraphicsContext, ck};
use crate::GL_TEXTURE_EXTERNAL_OES;
use alvr_common::glam::{IVec2, UVec2};
use glow::{self as gl, HasContext};
use std::{ffi::c_void, rc::Rc};

fn create_program(
    gl: &gl::Context,
    vertex_shader_source: &str,
    fragment_shader_source: &str,
) -> gl::Program {
    unsafe {
        let vertex_shader = ck!(gl.create_shader(gl::VERTEX_SHADER).unwrap());
        ck!(gl.shader_source(vertex_shader, vertex_shader_source));
        ck!(gl.compile_shader(vertex_shader));
        if !gl.get_shader_compile_status(vertex_shader) {
            panic!(
                "Failed to compile vertex shader: {}",
                gl.get_shader_info_log(vertex_shader)
            );
        }

        let fragment_shader = ck!(gl.create_shader(gl::FRAGMENT_SHADER).unwrap());
        ck!(gl.shader_source(fragment_shader, fragment_shader_source));
        ck!(gl.compile_shader(fragment_shader));
        if !gl.get_shader_compile_status(fragment_shader) {
            panic!(
                "Failed to compile fragment shader: {}",
                gl.get_shader_info_log(fragment_shader)
            );
        }

        let program = ck!(gl.create_program().unwrap());
        ck!(gl.attach_shader(program, vertex_shader));
        ck!(gl.attach_shader(program, fragment_shader));
        ck!(gl.link_program(program));
        if !gl.get_program_link_status(program) {
            panic!(
                "Failed to link program: {}",
                gl.get_program_info_log(program)
            );
        }

        ck!(gl.delete_shader(vertex_shader));
        ck!(gl.delete_shader(fragment_shader));

        program
    }
}

pub struct StagingRenderer {
    context: Rc<GraphicsContext>,
    program: gl::Program,
    view_idx_uloc: gl::UniformLocation,
    surface_texture: gl::Texture,
    viewport_size: IVec2,

    // CPU frame upload path (software decoding fallback)
    #[cfg(target_os = "linux")]
    program_2d: gl::Program,
    #[cfg(target_os = "linux")]
    view_idx_uloc_2d: gl::UniformLocation,
    #[cfg(target_os = "linux")]
    upload_texture: gl::Texture,
    #[cfg(target_os = "linux")]
    upload_texture_size: std::cell::Cell<IVec2>,
}

impl StagingRenderer {
    pub fn new(
        context: Rc<GraphicsContext>,
        view_resolution: UVec2,
        fix_limited_range: bool,
    ) -> Self {
        let gl = &context.gl_context;
        context.make_current();

        // Add #defines into the shader after the first line
        let mut frag_lines: Vec<&str> = include_str!("../resources/staging_fragment.glsl")
            .lines()
            .collect();
        if fix_limited_range {
            frag_lines.insert(1, "#line 0 1\n#define FIX_LIMITED_RANGE");
        }
        let frag_str = frag_lines.join("\n");

        let program = create_program(
            gl,
            include_str!("../resources/staging_vertex.glsl"),
            frag_str.as_str(),
        );

        #[cfg(target_os = "linux")]
        let program_2d = create_program(
            gl,
            include_str!("../resources/staging_vertex.glsl"),
            &frag_str
                .replace("#extension GL_OES_EGL_image_external_essl3 : enable", "")
                .replace("samplerExternalOES", "sampler2D"),
        );

        unsafe {
            // This is an external surface and storage should not be initialized
            let surface_texture = ck!(gl.create_texture().unwrap());

            let view_idx_uloc = ck!(gl.get_uniform_location(program, "view_idx")).unwrap();

            #[cfg(target_os = "linux")]
            let view_idx_uloc_2d = ck!(gl.get_uniform_location(program_2d, "view_idx")).unwrap();

            #[cfg(target_os = "linux")]
            let upload_texture = {
                let tex = ck!(gl.create_texture().unwrap());
                ck!(gl.bind_texture(gl::TEXTURE_2D, Some(tex)));
                ck!(gl.tex_parameter_i32(
                    gl::TEXTURE_2D,
                    gl::TEXTURE_MIN_FILTER,
                    gl::LINEAR as i32
                ));
                ck!(gl.tex_parameter_i32(
                    gl::TEXTURE_2D,
                    gl::TEXTURE_MAG_FILTER,
                    gl::LINEAR as i32
                ));
                ck!(gl.tex_parameter_i32(
                    gl::TEXTURE_2D,
                    gl::TEXTURE_WRAP_S,
                    gl::CLAMP_TO_EDGE as i32
                ));
                ck!(gl.tex_parameter_i32(
                    gl::TEXTURE_2D,
                    gl::TEXTURE_WRAP_T,
                    gl::CLAMP_TO_EDGE as i32
                ));
                ck!(gl.bind_texture(gl::TEXTURE_2D, None));
                tex
            };

            Self {
                context,
                program,
                surface_texture,
                view_idx_uloc,
                viewport_size: view_resolution.as_ivec2(),
                #[cfg(target_os = "linux")]
                program_2d,
                #[cfg(target_os = "linux")]
                view_idx_uloc_2d,
                #[cfg(target_os = "linux")]
                upload_texture,
                #[cfg(target_os = "linux")]
                upload_texture_size: std::cell::Cell::new(IVec2::ZERO),
            }
        }
    }

    pub fn create_framebuffers(&self, staging_textures: &[gl::Texture; 2]) -> [gl::Framebuffer; 2] {
        let gl = &self.context.gl_context;
        self.context.make_current();

        unsafe {
            let mut framebuffers = vec![];
            for &tex in staging_textures {
                let framebuffer = ck!(gl.create_framebuffer().unwrap());
                ck!(gl.bind_framebuffer(gl::DRAW_FRAMEBUFFER, Some(framebuffer)));
                ck!(gl.framebuffer_texture_2d(
                    gl::DRAW_FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::TEXTURE_2D,
                    Some(tex),
                    0,
                ));

                framebuffers.push(framebuffer);
            }

            ck!(gl.bind_framebuffer(gl::FRAMEBUFFER, None));

            framebuffers.try_into().unwrap()
        }
    }

    // Draw the bound source texture into both per-eye staging framebuffers
    fn draw_views(
        &self,
        framebuffers: &[gl::Framebuffer; 2],
        program: gl::Program,
        view_idx_uloc: &gl::UniformLocation,
        texture_target: u32,
        texture: gl::Texture,
    ) {
        let gl = &self.context.gl_context;

        unsafe {
            ck!(gl.use_program(Some(program)));

            ck!(gl.viewport(0, 0, self.viewport_size.x, self.viewport_size.y));
            ck!(gl.disable(gl::SCISSOR_TEST));
            ck!(gl.disable(gl::STENCIL_TEST));

            for (i, framebuffer) in framebuffers.iter().enumerate() {
                ck!(gl.bind_framebuffer(gl::DRAW_FRAMEBUFFER, Some(*framebuffer)));

                ck!(gl.active_texture(gl::TEXTURE0));
                ck!(gl.bind_texture(texture_target, Some(texture)));
                ck!(gl.bind_sampler(0, None));
                ck!(gl.uniform_1_i32(Some(view_idx_uloc), i as i32));
                ck!(gl.draw_arrays(gl::TRIANGLE_STRIP, 0, 4));
            }
        }
    }

    /// # Safety
    /// On Android, `hardware_buffer` must be a valid AHardwareBuffer. On Linux, it must point to a
    /// valid `LinuxVideoFrame`.
    #[allow(unused_variables)]
    pub fn render(&self, framebuffers: &[gl::Framebuffer; 2], hardware_buffer: *mut c_void) {
        self.context.make_current();

        #[cfg(target_os = "linux")]
        {
            if hardware_buffer.is_null() {
                return;
            }

            match unsafe { &*(hardware_buffer as *const crate::LinuxVideoFrame) } {
                crate::LinuxVideoFrame::DmaBuf(image) => unsafe {
                    self.context
                        .render_dmabuf_using_texture(image, self.surface_texture, || {
                            self.draw_views(
                                framebuffers,
                                self.program,
                                &self.view_idx_uloc,
                                GL_TEXTURE_EXTERNAL_OES,
                                self.surface_texture,
                            )
                        });
                },
                crate::LinuxVideoFrame::Rgba {
                    width,
                    height,
                    data,
                } => {
                    self.upload_rgba(*width as i32, *height as i32, data);
                    self.draw_views(
                        framebuffers,
                        self.program_2d,
                        &self.view_idx_uloc_2d,
                        gl::TEXTURE_2D,
                        self.upload_texture,
                    );
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        unsafe {
            self.context.render_ahardwarebuffer_using_texture(
                hardware_buffer,
                self.surface_texture,
                || {
                    self.draw_views(
                        framebuffers,
                        self.program,
                        &self.view_idx_uloc,
                        GL_TEXTURE_EXTERNAL_OES,
                        self.surface_texture,
                    )
                },
            )
        };
    }

    #[cfg(target_os = "linux")]
    fn upload_rgba(&self, width: i32, height: i32, data: &[u8]) {
        let gl = &self.context.gl_context;

        if data.len() < (width * height * 4) as usize {
            alvr_common::error!("RGBA frame buffer too small");
            return;
        }

        unsafe {
            ck!(gl.bind_texture(gl::TEXTURE_2D, Some(self.upload_texture)));
            ck!(gl.pixel_store_i32(gl::UNPACK_ALIGNMENT, 4));
            ck!(gl.pixel_store_i32(gl::UNPACK_ROW_LENGTH, 0));

            let size = IVec2::new(width, height);
            if self.upload_texture_size.get() != size {
                ck!(gl.tex_image_2d(
                    gl::TEXTURE_2D,
                    0,
                    gl::RGBA8 as i32,
                    width,
                    height,
                    0,
                    gl::RGBA,
                    gl::UNSIGNED_BYTE,
                    gl::PixelUnpackData::Slice(Some(data)),
                ));
                self.upload_texture_size.set(size);
            } else {
                ck!(gl.tex_sub_image_2d(
                    gl::TEXTURE_2D,
                    0,
                    0,
                    0,
                    width,
                    height,
                    gl::RGBA,
                    gl::UNSIGNED_BYTE,
                    gl::PixelUnpackData::Slice(Some(data)),
                ));
            }
        }
    }
}

impl Drop for StagingRenderer {
    fn drop(&mut self) {
        let gl = &self.context.gl_context;
        self.context.make_current();

        unsafe {
            ck!(gl.delete_program(self.program));
            ck!(gl.delete_texture(self.surface_texture));

            #[cfg(target_os = "linux")]
            {
                ck!(gl.delete_program(self.program_2d));
                ck!(gl.delete_texture(self.upload_texture));
            }
        }
    }
}
