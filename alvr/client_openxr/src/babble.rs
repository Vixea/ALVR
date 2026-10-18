use alvr_common::{
    anyhow::Result,
    parking_lot::Mutex,
    warn,
};
use rosc::{OscMessage, OscPacket, OscType};
use std::{
    collections::HashMap,
    net::UdpSocket,
    time::{Duration, Instant},
};

const BABBLE_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Default)]
struct BabbleState {
    last_received: Option<Instant>,
    // Eye lids / openness (1.0 = open, 0.0 = closed)
    left_eye_lid: Option<f32>,
    right_eye_lid: Option<f32>,
    left_eye_widen: Option<f32>,
    right_eye_widen: Option<f32>,
    left_eye_squint: Option<f32>,
    right_eye_squint: Option<f32>,
    left_eye_brow: Option<f32>,
    right_eye_brow: Option<f32>,

    // Mouth / Face blendshapes
    shapes: HashMap<String, f32>,
}

pub struct BabbleReceiver {
    socket: UdpSocket,
    state: Mutex<BabbleState>,
}

impl BabbleReceiver {
    pub fn new(port: u16) -> Result<Self> {
        let socket = UdpSocket::bind(format!("127.0.0.1:{port}"))?;
        socket.set_nonblocking(true)?;

        Ok(Self {
            socket,
            state: Mutex::new(BabbleState::default()),
        })
    }

    pub fn get_face_expressions(&self) -> Option<Vec<f32>> {
        let mut buffer = [0u8; 2048];
        let mut state = self.state.lock();

        loop {
            match self.socket.recv(&mut buffer) {
                Ok(size) => {
                    if let Ok((_, packet)) = rosc::decoder::decode_udp(&buffer[..size]) {
                        Self::process_packet(packet, &mut state);
                        state.last_received = Some(Instant::now());
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => {
                    warn!("Babble socket receive error: {e}");
                    break;
                }
            }
        }

        let last_received = state.last_received?;
        if last_received.elapsed() > BABBLE_TIMEOUT {
            return None;
        }

        Some(Self::state_to_fb_weights(&state))
    }

    fn process_packet(packet: OscPacket, state: &mut BabbleState) {
        match packet {
            OscPacket::Message(msg) => Self::process_message(msg, state),
            OscPacket::Bundle(bundle) => {
                for p in bundle.content {
                    Self::process_packet(p, state);
                }
            }
        }
    }

    fn process_message(msg: OscMessage, state: &mut BabbleState) {
        let name = msg.addr.rsplit('/').next().unwrap_or(&msg.addr);
        let val = match msg.args.first() {
            Some(OscType::Float(f)) => *f,
            Some(OscType::Double(d)) => *d as f32,
            Some(OscType::Int(i)) => *i as f32,
            _ => return,
        };

        match name {
            "LeftEyeLid" => state.left_eye_lid = Some(val),
            "RightEyeLid" => state.right_eye_lid = Some(val),
            "LeftEyeWiden" => state.left_eye_widen = Some(val),
            "RightEyeWiden" => state.right_eye_widen = Some(val),
            "LeftEyeSquint" => state.left_eye_squint = Some(val),
            "RightEyeSquint" => state.right_eye_squint = Some(val),
            "LeftEyeBrow" => state.left_eye_brow = Some(val),
            "RightEyeBrow" => state.right_eye_brow = Some(val),
            "CombinedEyeWiden" => {
                state.left_eye_widen = Some(val);
                state.right_eye_widen = Some(val);
            }
            "CombinedEyeSquint" => {
                state.left_eye_squint = Some(val);
                state.right_eye_squint = Some(val);
            }
            "CombinedEyeBrow" => {
                state.left_eye_brow = Some(val);
                state.right_eye_brow = Some(val);
            }
            other => {
                state.shapes.insert(other.to_string(), val);
            }
        }
    }

    fn state_to_fb_weights(state: &BabbleState) -> Vec<f32> {
        let mut weights = vec![0.0f32; 70];

        let get = |name: &str| -> f32 {
            state.shapes.get(name).copied().unwrap_or(0.0).clamp(0.0, 1.0)
        };

        // Brows
        weights[0] = state.left_eye_brow.unwrap_or(0.0).clamp(0.0, 1.0); // BROW_LOWERER_L
        weights[1] = state.right_eye_brow.unwrap_or(0.0).clamp(0.0, 1.0); // BROW_LOWERER_R

        // Cheeks
        weights[2] = get("cheekPuffLeft"); // CHEEK_PUFF_L
        weights[3] = get("cheekPuffRight"); // CHEEK_PUFF_R
        weights[6] = get("cheekSuckLeft"); // CHEEK_SUCK_L
        weights[7] = get("cheekSuckRight"); // CHEEK_SUCK_R

        // Chin
        weights[8] = get("mouthShrugLower"); // CHIN_RAISER_B
        weights[9] = get("mouthShrugUpper"); // CHIN_RAISER_T

        // Dimples
        weights[10] = get("mouthDimpleLeft"); // DIMPLER_L
        weights[11] = get("mouthDimpleRight"); // DIMPLER_R

        // Eyes closed (Babble Lid is 1.0 = open, 0.0 = closed; FB closed is 1.0 = closed)
        if let Some(lid) = state.left_eye_lid {
            weights[12] = (1.0 - lid).clamp(0.0, 1.0);
        }
        if let Some(lid) = state.right_eye_lid {
            weights[13] = (1.0 - lid).clamp(0.0, 1.0);
        }

        // Jaw
        weights[24] = get("jawOpen"); // JAW_DROP
        weights[25] = get("jawLeft"); // JAW_SIDEWAYS_LEFT
        weights[26] = get("jawRight"); // JAW_SIDEWAYS_RIGHT
        weights[27] = get("jawForward"); // JAW_THRUST

        // Lid tightener (squint)
        if let Some(squint) = state.left_eye_squint {
            weights[28] = squint.clamp(0.0, 1.0);
        }
        if let Some(squint) = state.right_eye_squint {
            weights[29] = squint.clamp(0.0, 1.0);
        }

        // Mouth corners (frown, smile)
        weights[30] = get("mouthFrownLeft"); // LIP_CORNER_DEPRESSOR_L
        weights[31] = get("mouthFrownRight"); // LIP_CORNER_DEPRESSOR_R
        weights[32] = get("mouthSmileLeft"); // LIP_CORNER_PULLER_L
        weights[33] = get("mouthSmileRight"); // LIP_CORNER_PULLER_R

        // Lip funnel
        let funnel = get("mouthFunnel");
        weights[34] = funnel; // LIP_FUNNELER_LB
        weights[35] = funnel; // LIP_FUNNELER_LT
        weights[36] = funnel; // LIP_FUNNELER_RB
        weights[37] = funnel; // LIP_FUNNELER_RT

        // Lip pressor
        weights[38] = get("mouthPressLeft"); // LIP_PRESSOR_L
        weights[39] = get("mouthPressRight"); // LIP_PRESSOR_R

        // Lip pucker
        let pucker = get("mouthPucker");
        weights[40] = pucker; // LIP_PUCKER_L
        weights[41] = pucker; // LIP_PUCKER_R

        // Lip stretcher
        weights[42] = get("mouthStretchLeft"); // LIP_STRETCHER_L
        weights[43] = get("mouthStretchRight"); // LIP_STRETCHER_R

        // Lip roll / suck
        let roll_lower = get("mouthRollLower");
        let roll_upper = get("mouthRollUpper");
        weights[44] = roll_lower; // LIP_SUCK_LB
        weights[45] = roll_upper; // LIP_SUCK_LT
        weights[46] = roll_lower; // LIP_SUCK_RB
        weights[47] = roll_upper; // LIP_SUCK_RT

        // Lips toward (mouth close)
        weights[50] = get("mouthClose"); // LIPS_TOWARD

        // Lip depressors
        weights[51] = get("mouthLowerDownLeft"); // LOWER_LIP_DEPRESSOR_L
        weights[52] = get("mouthLowerDownRight"); // LOWER_LIP_DEPRESSOR_R

        // Mouth sideways
        weights[53] = get("mouthLeft"); // MOUTH_LEFT
        weights[54] = get("mouthRight"); // MOUTH_RIGHT

        // Nose sneer
        weights[55] = get("noseSneerLeft"); // NOSE_WRINKLER_L
        weights[56] = get("noseSneerRight"); // NOSE_WRINKLER_R

        // Eye widen
        if let Some(widen) = state.left_eye_widen {
            weights[59] = widen.clamp(0.0, 1.0); // UPPER_LID_RAISER_L
        }
        if let Some(widen) = state.right_eye_widen {
            weights[60] = widen.clamp(0.0, 1.0); // UPPER_LID_RAISER_R
        }

        // Upper lip raiser
        weights[61] = get("mouthUpperUpLeft"); // UPPER_LIP_RAISER_L
        weights[62] = get("mouthUpperUpRight"); // UPPER_LIP_RAISER_R

        // Tongue
        let tongue_out = get("tongueOut");
        weights[63] = tongue_out; // TONGUE_TIP_INTERDENTAL
        weights[64] = get("tongueUp"); // TONGUE_TIP_ALVEOLAR
        weights[65] = get("tongueDown"); // FRONT_DORSAL_PALATE
        weights[66] = get("tongueRoll"); // MID_DORSAL_PALATE
        weights[68] = tongue_out; // TONGUE_OUT

        weights
    }
}
