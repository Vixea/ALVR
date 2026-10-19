use alvr_common::{anyhow::Result, glam::EulerRot, info, warn};
use alvr_packets::{FaceData, FaceExpressions};
use alvr_session::FaceTrackingSocialPresenceSinkConfig;
use rosc::{OscMessage, OscPacket, OscType};
use std::{
    f32::consts::PI,
    net::UdpSocket,
    time::{Duration, Instant},
};

const RAD_TO_DEG: f32 = 180.0 / PI;

const VRCFT_PORT: u16 = 0xA1F7;

pub struct FaceTrackingSink {
    config: FaceTrackingSocialPresenceSinkConfig,
    socket: UdpSocket,
    packet_buffer: Vec<u8>,
    last_log: Instant,
    packets_sent: u64,
}

impl FaceTrackingSink {
    pub fn new(config: FaceTrackingSocialPresenceSinkConfig, local_osc_port: u16) -> Result<Self> {
        let port = match config {
            FaceTrackingSocialPresenceSinkConfig::VrchatEyeOsc { port } => port,
            FaceTrackingSocialPresenceSinkConfig::VrcFaceTracking => VRCFT_PORT,
        };

        let socket = UdpSocket::bind(format!("127.0.0.1:{local_osc_port}"))
            .or_else(|_| UdpSocket::bind("127.0.0.1:0"))?;
        socket.connect(format!("127.0.0.1:{port}"))?;

        info!("Face tracking sink connected to 127.0.0.1:{port}");

        Ok(Self {
            config,
            socket,
            packet_buffer: vec![],
            last_log: Instant::now(),
            packets_sent: 0,
        })
    }

    fn send_osc_message(&mut self, path: &str, args: Vec<OscType>) {
        if self
            .socket
            .send(
                &rosc::encoder::encode(&OscPacket::Message(OscMessage {
                    addr: path.into(),
                    args,
                }))
                .unwrap(),
            )
            .is_ok()
        {
            self.packets_sent += 1;
        }
    }

    fn append_packet_vrcft(&mut self, prefix: [u8; 8], data: &[f32]) {
        self.packet_buffer.extend(prefix);

        for val in data {
            self.packet_buffer.extend(val.to_le_bytes());
        }
    }

    pub fn send_tracking(&mut self, face_data: &FaceData) {
        match self.config {
            FaceTrackingSocialPresenceSinkConfig::VrchatEyeOsc { .. } => {
                if let [Some(left), Some(right)] = face_data.eyes_social {
                    let (left_pitch, left_yaw, _) = left.to_euler(EulerRot::XYZ);
                    let (right_pitch, right_yaw, _) = right.to_euler(EulerRot::XYZ);

                    self.send_osc_message(
                        "/tracking/eye/LeftRightPitchYaw",
                        vec![
                            OscType::Float(-left_pitch * RAD_TO_DEG),
                            OscType::Float(-left_yaw * RAD_TO_DEG),
                            OscType::Float(-right_pitch * RAD_TO_DEG),
                            OscType::Float(-right_yaw * RAD_TO_DEG),
                        ],
                    );
                } else if let Some(quat) = face_data.eyes_combined {
                    let (pitch, yaw, _) = quat.to_euler(EulerRot::XYZ);

                    self.send_osc_message(
                        "/tracking/eye/CenterPitchYaw",
                        vec![
                            OscType::Float(-pitch * RAD_TO_DEG),
                            OscType::Float(-yaw * RAD_TO_DEG),
                        ],
                    );
                }

                let (left_eye_blink, right_eye_blink) = match &face_data.face_expressions {
                    Some(FaceExpressions::Fb(items)) => (Some(items[12]), Some(items[13])),
                    Some(FaceExpressions::Bd(items)) => (Some(items[28]), Some(items[38])),
                    Some(FaceExpressions::Htc { eye, .. }) => {
                        (eye.as_ref().map(|v| v[0]), eye.as_ref().map(|v| v[2]))
                    }
                    _ => (None, None),
                };

                if let (Some(left), Some(right)) = (left_eye_blink, right_eye_blink) {
                    self.send_osc_message(
                        "/tracking/eye/EyesClosedAmount",
                        vec![OscType::Float((left + right) / 2.0)],
                    );
                } else if let Some(blink) = left_eye_blink.or(right_eye_blink) {
                    self.send_osc_message(
                        "/tracking/eye/EyesClosedAmount",
                        vec![OscType::Float(blink)],
                    );
                }

                if self.last_log.elapsed() >= Duration::from_secs(5) {
                    if self.packets_sent > 0 {
                        info!("VRChat Eye OSC sink: {} packets sent in last 5s", self.packets_sent);
                    }
                    self.packets_sent = 0;
                    self.last_log = Instant::now();
                }
            }
            FaceTrackingSocialPresenceSinkConfig::VrcFaceTracking => {
                self.packet_buffer.clear();

                if let [Some(left_quat), Some(right_quat)] = face_data.eyes_social {
                    let mut vec = left_quat.to_array().to_vec();
                    vec.extend_from_slice(&right_quat.to_array());
                    self.append_packet_vrcft(*b"EyesQuat", &vec);
                } else if let Some(quat) = face_data.eyes_combined {
                    self.append_packet_vrcft(*b"CombQuat", &quat.to_array());
                }

                let expr_tag = match &face_data.face_expressions {
                    Some(FaceExpressions::Fb(items)) => {
                        self.append_packet_vrcft(*b"Face2Fb\0", items);
                        if items.len() >= 14 {
                            format!("Fb (EyesClosed L={:.2}, R={:.2})", items[12], items[13])
                        } else {
                            "Fb".to_string()
                        }
                    }
                    Some(FaceExpressions::Bd(items)) => {
                        self.append_packet_vrcft(*b"FacePico", items);
                        "Bd".to_string()
                    }
                    Some(FaceExpressions::Htc { eye, lip }) => {
                        if let Some(arr) = eye {
                            self.append_packet_vrcft(*b"EyesHtc\0", arr);
                        }

                        if let Some(arr) = lip {
                            self.append_packet_vrcft(*b"LipHtc\0\0", arr);
                        }
                        "Htc".to_string()
                    }
                    None => "None".to_string(),
                };

                if !self.packet_buffer.is_empty() {
                    if let Err(e) = self.socket.send(&self.packet_buffer) {
                        warn!("Failed to send tracking data to VRCFT: {e}");
                    } else {
                        self.packets_sent += 1;
                    }
                }

                if self.last_log.elapsed() >= Duration::from_secs(5) {
                    if self.packets_sent > 0 {
                        info!(
                            "VRCFT sink: {} packets sent in last 5s (expressions: {})",
                            self.packets_sent, expr_tag
                        );
                    } else {
                        warn!(
                            "VRCFT sink: 0 packets sent in last 5s (expressions: {})",
                            expr_tag
                        );
                    }
                    self.packets_sent = 0;
                    self.last_log = Instant::now();
                }
            }
        }
    }
}
