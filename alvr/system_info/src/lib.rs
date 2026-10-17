#[cfg(target_os = "android")]
pub mod android;

#[cfg(target_os = "android")]
pub use android::*;

use alvr_common::settings_schema::SettingsSchema;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

pub const PACKAGE_NAME_STORE: &str = "alvr.client";
pub const PACKAGE_NAME_GITHUB_DEV: &str = "alvr.client.dev";
pub const PACKAGE_NAME_GITHUB_STABLE: &str = "alvr.client.stable";

// Platform of the device. It is used to match the VR runtime and enable features conditionally.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Platform {
    Quest1,
    Quest2,
    Quest3,
    Quest3S,
    QuestPro,
    QuestUnknown,
    PicoNeo3,
    Pico4,
    Pico4Pro,
    Pico4Enterprise,
    Pico4Ultra,
    PicoG3,
    PicoUnknown,
    Focus3,
    FocusVision,
    XRElite,
    ViveUnknown,
    Yvr,
    PlayForDreamMR,
    SamsungGalaxyXR,
    AndroidUnknown,
    SteamFrame,
    VisionOSHeadset,
    WindowsPc,
    LinuxPc,
    Macos,
    Unknown,
}

impl Platform {
    pub const fn is_quest(&self) -> bool {
        matches!(
            self,
            Self::Quest1
                | Self::Quest2
                | Self::Quest3
                | Self::Quest3S
                | Self::QuestPro
                | Self::QuestUnknown
        )
    }

    pub const fn is_pico(&self) -> bool {
        matches!(
            self,
            Self::PicoG3
                | Self::PicoNeo3
                | Self::Pico4
                | Self::Pico4Pro
                | Self::Pico4Enterprise
                | Self::Pico4Ultra
                | Self::PicoUnknown
        )
    }

    pub const fn is_vive(&self) -> bool {
        matches!(
            self,
            Self::Focus3 | Self::FocusVision | Self::XRElite | Self::ViveUnknown
        )
    }

    pub const fn is_yvr(&self) -> bool {
        matches!(self, Self::Yvr | Self::PlayForDreamMR)
    }
}

impl Display for Platform {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Quest1 => "Quest 1",
            Self::Quest2 => "Quest 2",
            Self::Quest3 => "Quest 3",
            Self::Quest3S => "Quest 3S",
            Self::QuestPro => "Quest Pro",
            Self::QuestUnknown => "Quest (unknown)",
            Self::PicoNeo3 => "Pico Neo 3",
            Self::Pico4 => "Pico 4",
            Self::Pico4Pro => "Pico 4 Pro",
            Self::Pico4Enterprise => "Pico 4 Enterprise",
            Self::Pico4Ultra => "Pico 4 Ultra",
            Self::PicoG3 => "Pico G3",
            Self::PicoUnknown => "Pico (unknown)",
            Self::Focus3 => "VIVE Focus 3",
            Self::FocusVision => "VIVE Focus Vision",
            Self::XRElite => "VIVE XR Elite",
            Self::ViveUnknown => "HTC VIVE (unknown)",
            Self::Yvr => "YVR",
            Self::PlayForDreamMR => "Play For Dream MR",
            Self::SamsungGalaxyXR => "Samsung Galaxy XR",
            Self::AndroidUnknown => "Android (unknown)",
            Self::SteamFrame => "Steam Frame",
            Self::VisionOSHeadset => "visionOS Headset",
            Self::WindowsPc => "Windows PC",
            Self::LinuxPc => "Linux PC",
            Self::Macos => "macOS",
            Self::Unknown => "Unknown",
        };
        write!(f, "{name}")
    }
}

#[cfg_attr(
    not(any(target_os = "android", target_os = "linux")),
    expect(unused_variables)
)]
pub fn platform(runtime_name: Option<String>, runtime_version: Option<u64>) -> Platform {
    #[cfg(target_os = "android")]
    {
        let manufacturer = android::manufacturer_name();
        let model = android::model_name();
        let device = android::device_name();
        let product = android::product_name();

        // TODO: Better Android XR heuristic
        // (Maybe check runtime json for /system/lib64/libopenxr.google.so?)

        alvr_common::info!(
            "manufacturer: {manufacturer}, model: {model}, device: {device}, product: {product}, \
            runtime_name: {runtime_name:?}, runtime_version: {runtime_version:?}",
        );

        match (
            manufacturer.as_str(),
            model.as_str(),
            device.as_str(),
            product.as_str(),
        ) {
            ("Oculus", _, "monterey", _) => Platform::Quest1,
            ("Oculus", _, "hollywood", _) => Platform::Quest2,
            ("Oculus", _, "eureka", _) => Platform::Quest3,
            ("Oculus", _, "panther", _) => Platform::Quest3S,
            ("Oculus", _, "seacliff", _) => Platform::QuestPro,
            ("Oculus", _, _, _) => Platform::QuestUnknown,
            ("Pico", "Pico Neo 3" | "Pico Neo3 Link", _, _) => Platform::PicoNeo3,
            ("Pico", _, _, "PICO 4 Pro") => Platform::Pico4Pro,
            ("Pico", _, _, "PICO 4 Enterprise") => Platform::Pico4Enterprise,
            ("Pico", _, _, "PICO 4") => Platform::Pico4,
            ("Pico", _, _, "PICO 4 Ultra") => Platform::Pico4Ultra,
            ("Pico", _, _, "PICO G3") => Platform::PicoG3,
            ("Pico", _, _, _) => Platform::PicoUnknown,
            ("HTC", "VIVE Focus 3", _, _) => Platform::Focus3,
            ("HTC", "VIVE Focus Vision", _, _) => Platform::FocusVision,
            ("HTC", "VIVE XR Series", _, _) => Platform::XRElite,
            ("HTC", _, _, _) => Platform::ViveUnknown,
            ("YVR", _, _, _) => Platform::Yvr,
            ("Play For Dream", _, _, _) => Platform::PlayForDreamMR,
            ("samsung", _, "xrvst2", _) => Platform::SamsungGalaxyXR,
            _ => Platform::AndroidUnknown,
        }
    }
    #[cfg(target_os = "linux")]
    {
        linux_platform(runtime_name, runtime_version)
    }
    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    {
        match std::env::consts::OS {
            "visionos" => Platform::VisionOSHeadset,
            "windows" => Platform::WindowsPc,
            "macos" => Platform::Macos,
            _ => Platform::Unknown,
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_platform(runtime_name: Option<String>, runtime_version: Option<u64>) -> Platform {
    // Note: these identifiers have not all been confirmed on retail hardware. "deckard" is the
    // Steam Frame codename.
    const STEAM_FRAME_IDS: &[&str] = &["steam frame", "steamframe", "deckard", "sm8650"];

    let read = |path: &str| {
        std::fs::read(path)
            .map(|bytes| {
                String::from_utf8_lossy(&bytes)
                    .replace('\0', " ")
                    .trim()
                    .to_lowercase()
            })
            .unwrap_or_default()
    };

    let dt_model = read("/sys/firmware/devicetree/base/model");
    let dt_compatible = read("/sys/firmware/devicetree/base/compatible");
    let dmi_product = read("/sys/devices/virtual/dmi/id/product_name");
    let dmi_board = read("/sys/devices/virtual/dmi/id/board_name");
    let runtime = runtime_name.clone().unwrap_or_default().to_lowercase();

    alvr_common::info!(
        "devicetree model: {dt_model:?}, compatible: {dt_compatible:?}, \
        dmi product: {dmi_product:?}, board: {dmi_board:?}, \
        runtime_name: {runtime_name:?}, runtime_version: {runtime_version:?}",
    );

    let is_steam_frame = [
        &dt_model,
        &dt_compatible,
        &dmi_product,
        &dmi_board,
        &runtime,
    ]
    .iter()
    .any(|s| STEAM_FRAME_IDS.iter().any(|id| s.contains(id)));

    // There are currently no other supported standalone Linux ARM headsets, so treat any
    // aarch64 Linux device as a Steam Frame.
    if is_steam_frame || cfg!(target_arch = "aarch64") {
        Platform::SteamFrame
    } else {
        Platform::LinuxPc
    }
}

#[cfg(not(target_os = "android"))]
pub fn local_ip() -> std::net::IpAddr {
    use std::net::{IpAddr, Ipv4Addr};

    local_ip_address::local_ip().unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
}

#[derive(SettingsSchema, Serialize, Deserialize, Clone)]
pub enum ClientFlavor {
    Store,
    Github,
    Custom(String),
}
