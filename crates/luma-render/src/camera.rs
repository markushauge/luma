use bevy::prelude::*;

#[derive(Component)]
pub struct Camera {
    pub sensor: Sensor,
    pub focal_length: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            sensor: Sensor::FULL_FRAME,
            focal_length: 24.0,
        }
    }
}

pub struct Sensor {
    pub width: f32,
    pub height: f32,
}

impl Sensor {
    /// 35mm full-frame sensor
    pub const FULL_FRAME: Self = Self {
        width: 36.0,
        height: 24.0,
    };

    pub fn horizontal_fov(&self, focal_length: f32) -> f32 {
        2.0 * (self.width / (2.0 * focal_length)).atan()
    }

    pub fn vertical_fov(&self, focal_length: f32) -> f32 {
        2.0 * (self.height / (2.0 * focal_length)).atan()
    }
}
