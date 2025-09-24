// Based on bevy_flycam

use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    window::{CursorGrabMode, PrimaryWindow},
};

// TODO: Make configurable
const MOVE_SENSITIVITY: f32 = 10.0;
const LOOK_SENSITIVITY: f32 = 0.0001;

const GAMEPAD_DEADZONE: f32 = 0.1;
const GAMEPAD_MOVE_SENSITIVITY: f32 = 5.0;
const GAMEPAD_LOOK_SENSITIVITY: f32 = 100.0;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (update_transform, update_transform_gamepad))
            .add_systems(Update, update_cursor_grab);
    }
}

#[derive(Component)]
pub struct Camera;

fn update_transform(
    window: Query<&Window, With<PrimaryWindow>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut state: EventReader<MouseMotion>,
    mut query: Query<&mut Transform, With<Camera>>,
) {
    let Ok(window) = window.single() else {
        return;
    };

    let Ok(mut transform) = query.single_mut() else {
        return;
    };

    if window.cursor_options.grab_mode != CursorGrabMode::Confined {
        return;
    }

    let mut velocity = Vec3::ZERO;
    let local_z = transform.local_z();
    let forward = -local_z.as_vec3();
    let right = forward.cross(Vec3::Y);

    for key in keys.get_pressed() {
        match key {
            KeyCode::KeyW => {
                velocity += forward;
            }
            KeyCode::KeyS => {
                velocity -= forward;
            }
            KeyCode::KeyA => {
                velocity -= right;
            }
            KeyCode::KeyD => {
                velocity += right;
            }
            KeyCode::Space => {
                velocity += Vec3::Y;
            }
            KeyCode::ShiftLeft => {
                velocity -= Vec3::Y;
            }
            _ => {}
        }
    }

    velocity = velocity.normalize_or_zero();
    transform.translation += velocity * time.delta_secs() * MOVE_SENSITIVITY;

    for event in state.read() {
        let (mut yaw, mut pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
        let window_scale = window.height().min(window.width());
        pitch -= (LOOK_SENSITIVITY * event.delta.y * window_scale).to_radians();
        yaw -= (LOOK_SENSITIVITY * event.delta.x * window_scale).to_radians();
        pitch = pitch.clamp(-1.54, 1.54);
        transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }
}

fn update_transform_gamepad(
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Camera>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };

    for gamepad in gamepads.iter() {
        let mut velocity = Vec3::ZERO;
        let local_z = transform.local_z();
        let forward = -local_z.as_vec3();
        let right = forward.cross(Vec3::Y);

        if let (Some(x), Some(y)) = (
            gamepad.get(GamepadAxis::LeftStickX),
            gamepad.get(GamepadAxis::LeftStickY),
        ) {
            if x.abs() > GAMEPAD_DEADZONE {
                velocity += right * x;
            }

            if y.abs() > GAMEPAD_DEADZONE {
                velocity += forward * y;
            }
        }

        if gamepad.pressed(GamepadButton::LeftTrigger2) {
            velocity += Vec3::Y;
        }

        if gamepad.pressed(GamepadButton::RightTrigger2) {
            velocity -= Vec3::Y;
        }

        transform.translation += velocity * time.delta_secs() * GAMEPAD_MOVE_SENSITIVITY;

        let (mut yaw, mut pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);

        if let (Some(x), Some(y)) = (
            gamepad.get(GamepadAxis::RightStickX),
            gamepad.get(GamepadAxis::RightStickY),
        ) {
            if x.abs() > GAMEPAD_DEADZONE {
                yaw -= (GAMEPAD_LOOK_SENSITIVITY * x * time.delta_secs()).to_radians();
            }

            if y.abs() > GAMEPAD_DEADZONE {
                pitch += (GAMEPAD_LOOK_SENSITIVITY * y * time.delta_secs()).to_radians();
            }
        }

        pitch = pitch.clamp(-1.54, 1.54);
        transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }
}

fn update_cursor_grab(
    mut window: Query<&mut Window, With<PrimaryWindow>>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    let Ok(mut window) = window.single_mut() else {
        return;
    };

    if mouse.just_pressed(MouseButton::Left) {
        window.cursor_options.grab_mode = CursorGrabMode::Confined;
        window.cursor_options.visible = false;
    }

    if keys.just_pressed(KeyCode::Escape) {
        window.cursor_options.grab_mode = CursorGrabMode::None;
        window.cursor_options.visible = true;
    }
}
