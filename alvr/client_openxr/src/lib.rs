mod graphics;
mod interaction;
mod lobby;
mod stream;

use crate::stream::StreamConfig;
use alvr_client_core::{ClientCapabilities, ClientCoreContext, ClientCoreEvent, Platform};
use alvr_common::{
    error,
    glam::{Quat, UVec2, Vec2, Vec3},
    info,
    parking_lot::RwLock,
    warn, Fov, Pose, HAND_LEFT_ID,
};
use lobby::Lobby;
use openxr as xr;
use std::{
    path::Path,
    ptr,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use stream::StreamContext;

const DECODER_MAX_TIMEOUT_MULTIPLIER: f32 = 0.8;

fn to_vec3(v: xr::Vector3f) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

fn to_quat(q: xr::Quaternionf) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

fn to_pose(p: xr::Posef) -> Pose {
    Pose {
        orientation: to_quat(p.orientation),
        position: to_vec3(p.position),
    }
}

fn to_fov(f: xr::Fovf) -> Fov {
    Fov {
        left: f.angle_left,
        right: f.angle_right,
        up: f.angle_up,
        down: f.angle_down,
    }
}

fn to_xr_time(timestamp: Duration) -> xr::Time {
    xr::Time::from_nanos(timestamp.as_nanos() as _)
}

#[derive(Clone)]
pub struct XrContext {
    instance: xr::Instance,
    system: xr::SystemId,
    session: xr::Session<xr::OpenGlEs>,
}

pub struct SessionRunningContext {
    reference_space: Arc<RwLock<xr::Space>>,
    lobby: Lobby,
    stream_context: Option<StreamContext>,
}

fn default_view() -> xr::View {
    xr::View {
        pose: xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: xr::Vector3f::default(),
        },
        fov: xr::Fovf {
            angle_left: -1.0,
            angle_right: 1.0,
            angle_up: 1.0,
            angle_down: -1.0,
        },
    }
}

pub fn entry_point() {
    alvr_client_core::init_logging();

    let platform = alvr_client_core::platform();

    let loader_suffix = match platform {
        Platform::Quest1
        | Platform::Quest2
        | Platform::Quest3
        | Platform::QuestPro
        | Platform::QuestUnknown => "quest",
        Platform::PicoNeo3 | Platform::Pico4 => "pico",
        Platform::Yvr => "yvr",
        Platform::Lynx => "lynx",
        _ => "generic",
    };
    let xr_entry = unsafe {
        xr::Entry::load_from(Path::new(&format!("libopenxr_loader_{loader_suffix}.so"))).unwrap()
    };

    #[cfg(target_os = "android")]
    xr_entry.initialize_android_loader().unwrap();

    let available_extensions = xr_entry.enumerate_extensions().unwrap();

    // todo: switch to vulkan
    assert!(available_extensions.khr_opengl_es_enable);

    let mut exts = xr::ExtensionSet::default();
    exts.bd_controller_interaction = available_extensions.bd_controller_interaction;
    exts.ext_eye_gaze_interaction = available_extensions.ext_eye_gaze_interaction;
    exts.ext_hand_tracking = available_extensions.ext_hand_tracking;
    exts.fb_color_space = available_extensions.fb_color_space;
    exts.fb_display_refresh_rate = available_extensions.fb_display_refresh_rate;
    exts.fb_eye_tracking_social = available_extensions.fb_eye_tracking_social;
    exts.fb_face_tracking2 = available_extensions.fb_face_tracking2;
    exts.fb_body_tracking = available_extensions.fb_body_tracking;
    exts.meta_body_tracking_full_body = available_extensions.meta_body_tracking_full_body;
    exts.fb_foveation = available_extensions.fb_foveation;
    exts.fb_foveation_configuration = available_extensions.fb_foveation_configuration;
    exts.fb_swapchain_update_state = available_extensions.fb_swapchain_update_state;
    exts.htc_facial_tracking = available_extensions.htc_facial_tracking;
    exts.htc_vive_focus3_controller_interaction =
        available_extensions.htc_vive_focus3_controller_interaction;
    #[cfg(target_os = "android")]
    {
        exts.khr_android_create_instance = true;
    }
    exts.khr_convert_timespec_time = true;
    exts.khr_opengl_es_enable = true;

    let xr_instance = xr_entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "ALVR Client",
                application_version: 0,
                engine_name: "ALVR",
                engine_version: 0,
            },
            &exts,
            &[],
        )
        .unwrap();

    let egl_context = graphics::init_egl();

    let mut last_lobby_message = String::new();
    let mut stream_config = None::<StreamConfig>;

    'session_loop: loop {
        let xr_system = xr_instance
            .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
            .unwrap();

        // mandatory call
        let _ = xr_instance
            .graphics_requirements::<xr::OpenGlEs>(xr_system)
            .unwrap();

        let (xr_session, mut xr_frame_waiter, mut xr_frame_stream) = unsafe {
            xr_instance
                .create_session(xr_system, &egl_context.session_create_info())
                .unwrap()
        };

        let xr_ctx = XrContext {
            instance: xr_instance.clone(),
            system: xr_system,
            session: xr_session.clone(),
        };

        let views_config = xr_instance
            .enumerate_view_configuration_views(
                xr_system,
                xr::ViewConfigurationType::PRIMARY_STEREO,
            )
            .unwrap();
        assert_eq!(views_config.len(), 2);

        let default_view_resolution = UVec2::new(
            views_config[0].recommended_image_rect_width,
            views_config[0].recommended_image_rect_height,
        );

        let refresh_rates = if exts.fb_display_refresh_rate {
            xr_session.enumerate_display_refresh_rates().unwrap()
        } else {
            vec![90.0]
        };

        let capabilities = ClientCapabilities {
            default_view_resolution,
            external_decoder: false,
            refresh_rates,
            foveated_encoding: platform != Platform::Unknown,
            encoder_high_profile: platform != Platform::Unknown,
            encoder_10_bits: platform != Platform::Unknown,
            encoder_av1: platform == Platform::Quest3,
        };
        let core_context = Arc::new(ClientCoreContext::new(capabilities));

        alvr_client_core::opengl::initialize();
        alvr_client_core::opengl::update_hud_message(&last_lobby_message);

        let interaction_context = Arc::new(interaction::initialize_interaction(
            &xr_ctx,
            platform,
            stream_config
                .as_ref()
                .and_then(|c| c.face_sources_config.clone()),
            stream_config
                .as_ref()
                .and_then(|c| c.body_sources_config.clone()),
        ));

        let mut session_running_context = None;

        let mut event_storage = xr::EventDataBuffer::new();
        'render_loop: loop {
            while let Some(event) = xr_instance.poll_event(&mut event_storage).unwrap() {
                match event {
                    xr::Event::EventsLost(event) => {
                        error!("OpenXR: lost {} events!", event.lost_event_count());
                    }
                    xr::Event::InstanceLossPending(_) => break 'session_loop,
                    xr::Event::SessionStateChanged(event) => match event.state() {
                        xr::SessionState::READY => {
                            xr_session
                                .begin(xr::ViewConfigurationType::PRIMARY_STEREO)
                                .unwrap();

                            let reference_space = Arc::new(RwLock::new(
                                xr_session
                                    .create_reference_space(
                                        xr::ReferenceSpaceType::STAGE,
                                        xr::Posef::IDENTITY,
                                    )
                                    .unwrap(),
                            ));

                            let lobby = Lobby::new(
                                xr_session.clone(),
                                Arc::clone(&reference_space),
                                default_view_resolution,
                            );

                            session_running_context = Some(SessionRunningContext {
                                reference_space,
                                lobby,
                                stream_context: None,
                            });

                            core_context.resume();
                        }
                        xr::SessionState::STOPPING => {
                            core_context.pause();
                            alvr_client_core::opengl::pause();

                            // Delete all resources and stop thread
                            session_running_context = None;

                            xr_session.end().unwrap();
                        }
                        xr::SessionState::EXITING => break 'render_loop,
                        xr::SessionState::LOSS_PENDING => break 'render_loop,
                        _ => (),
                    },
                    xr::Event::ReferenceSpaceChangePending(event) => {
                        info!(
                            "ReferenceSpaceChangePending type: {:?}",
                            event.reference_space_type()
                        );

                        if let Some(ctx) = &session_running_context {
                            *ctx.reference_space.write() = xr_session
                                .create_reference_space(
                                    xr::ReferenceSpaceType::STAGE,
                                    xr::Posef::IDENTITY,
                                )
                                .unwrap();

                            core_context.send_playspace(
                                xr_session
                                    .reference_space_bounds_rect(xr::ReferenceSpaceType::STAGE)
                                    .unwrap()
                                    .map(|a| Vec2::new(a.width, a.height)),
                            );
                        }
                    }
                    xr::Event::PerfSettingsEXT(event) => {
                        info!(
                            "Perf: from {:?} to {:?}, domain: {:?}/{:?}",
                            event.from_level(),
                            event.to_level(),
                            event.domain(),
                            event.sub_domain(),
                        );
                    }
                    xr::Event::InteractionProfileChanged(_) => {
                        // todo
                    }
                    xr::Event::PassthroughStateChangedFB(_) => {
                        // todo
                    }
                    _ => (),
                }
            }

            let session_context = if let Some(ctx) = &mut session_running_context {
                ctx
            } else {
                thread::sleep(Duration::from_millis(100));
                continue;
            };

            while let Some(event) = core_context.poll_event() {
                match event {
                    ClientCoreEvent::UpdateHudMessage(message) => {
                        last_lobby_message = message.clone();
                        alvr_client_core::opengl::update_hud_message(&message);
                    }
                    ClientCoreEvent::StreamingStarted {
                        settings,
                        negotiated_config,
                    } => {
                        let new_config = StreamConfig::new(&settings, negotiated_config);

                        // combined_eye_gaze is a setting that needs to be enabled at session
                        // creation. Since HTC headsets don't support session reinitialization, skip
                        // all elements that need it, that is face and eye tracking.
                        if stream_config.as_ref() != Some(&new_config)
                            && !matches!(
                                platform,
                                Platform::Focus3 | Platform::XRElite | Platform::ViveUnknown
                            )
                        {
                            stream_config = Some(new_config);

                            xr_session.request_exit().ok();
                            continue;
                        }

                        session_context.stream_context = Some(StreamContext::new(
                            Arc::clone(&core_context),
                            &xr_ctx,
                            Arc::clone(&interaction_context),
                            Arc::clone(&session_context.reference_space),
                            platform,
                            &new_config,
                        ));

                        stream_config = Some(new_config);
                    }
                    ClientCoreEvent::StreamingStopped => {
                        session_context.stream_context = None;
                    }
                    ClientCoreEvent::Haptics {
                        device_id,
                        duration,
                        frequency,
                        amplitude,
                    } => {
                        let action = if device_id == *HAND_LEFT_ID {
                            &interaction_context.hands_interaction[0].vibration_action
                        } else {
                            &interaction_context.hands_interaction[1].vibration_action
                        };

                        action
                            .apply_feedback(
                                &xr_session,
                                xr::Path::NULL,
                                &xr::HapticVibration::new()
                                    .amplitude(amplitude)
                                    .frequency(frequency)
                                    .duration(xr::Duration::from_nanos(duration.as_nanos() as _)),
                            )
                            .unwrap();
                    }
                    _ => panic!(),
                }
            }

            let frame_state = match xr_frame_waiter.wait() {
                Ok(state) => state,
                Err(e) => {
                    error!("{e}");
                    panic!();
                }
            };
            let frame_interval =
                Duration::from_nanos(frame_state.predicted_display_period.as_nanos() as _);
            let vsync_time =
                Duration::from_nanos(frame_state.predicted_display_time.as_nanos() as _);

            xr_frame_stream.begin().unwrap();

            if !frame_state.should_render {
                xr_frame_stream
                    .end(
                        frame_state.predicted_display_time,
                        xr::EnvironmentBlendMode::OPAQUE,
                        &[],
                    )
                    .unwrap();

                continue;
            }

            // todo: allow rendering lobby and stream layers at the same time and add cross fade
            let (layer, display_time) = if let Some(context) = &mut session_context.stream_context {
                let frame_poll_deadline = Instant::now()
                    + Duration::from_secs_f32(
                        frame_interval.as_secs_f32() * DECODER_MAX_TIMEOUT_MULTIPLIER,
                    );
                let mut frame_result = None;
                while frame_result.is_none() && Instant::now() < frame_poll_deadline {
                    frame_result = core_context.get_frame();
                    thread::yield_now();
                }

                let (timestamp, hardware_buffer) = if let Some(pair) = frame_result {
                    pair
                } else {
                    warn!("Timed out when waiting for frame!");
                    (vsync_time, ptr::null_mut())
                };

                let layer = context.render(timestamp, hardware_buffer, vsync_time);

                (layer, timestamp)
            } else {
                let layer = session_context
                    .lobby
                    .render(frame_state.predicted_display_time);

                (layer, vsync_time)
            };

            let res = xr_frame_stream.end(
                to_xr_time(display_time),
                xr::EnvironmentBlendMode::OPAQUE,
                &[&xr::CompositionLayerProjection::new()
                    .space(&session_context.reference_space.read())
                    .views(&layer)],
            );

            if let Err(e) = res {
                let time = to_xr_time(display_time);
                error!("End frame failed! {e}, timestamp: {display_time:?}, time: {time:?}");

                xr_frame_stream
                    .end(
                        frame_state.predicted_display_time,
                        xr::EnvironmentBlendMode::OPAQUE,
                        &[],
                    )
                    .unwrap();
            }
        }

        alvr_client_core::opengl::destroy();
    }
}

#[allow(unused)]
fn xr_runtime_now(xr_instance: &xr::Instance) -> Option<Duration> {
    let time_nanos = xr_instance.now().ok()?.as_nanos();

    (time_nanos > 0).then(|| Duration::from_nanos(time_nanos as _))
}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: android_activity::AndroidApp) {
    use android_activity::{InputStatus, MainEvent, PollEvent};

    let rendering_thread = thread::spawn(|| {
        // workaround for the Pico runtime
        let context = ndk_context::android_context();
        let vm = unsafe { jni::JavaVM::from_raw(context.vm().cast()) }.unwrap();
        let _env = vm.attach_current_thread().unwrap();

        entry_point();
    });

    let mut should_quit = false;
    while !should_quit {
        app.poll_events(Some(Duration::from_millis(100)), |event| match event {
            PollEvent::Main(MainEvent::Destroy) => {
                should_quit = true;
            }
            PollEvent::Main(MainEvent::InputAvailable) => {
                if let Ok(mut iter) = app.input_events_iter() {
                    while iter.next(|_| InputStatus::Unhandled) {}
                }
            }
            _ => (),
        });
    }

    // Note: the quit event is sent from OpenXR too, this will return rather quicly.
    rendering_thread.join().unwrap();
}
