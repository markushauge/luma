// Based on bevy_flycam

use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};

// TODO: Make configurable
const MOVE_SENSITIVITY: f32 = 10.0;
const LOOK_SENSITIVITY: f32 = 0.0001;

const GAMEPAD_JOYSTICK_DEADZONE: f32 = 0.1;
const GAMEPAD_TRIGGER_DEADZONE: f32 = 0.01;
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
    cursor_options: Query<&CursorOptions, With<PrimaryWindow>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut messages: MessageReader<MouseMotion>,
    mut query: Query<&mut Transform, With<Camera>>,
) {
    let Ok(window) = window.single() else {
        return;
    };

    let Ok(cursor_options) = cursor_options.single() else {
        return;
    };

    let Ok(mut transform) = query.single_mut() else {
        return;
    };

    if cursor_options.grab_mode != CursorGrabMode::Confined {
        return;
    }

    let mut velocity = Vec3::ZERO;
    let forward = -transform.local_z().as_vec3();
    let right = transform.local_x().as_vec3();

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

    for message in messages.read() {
        let (mut yaw, mut pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
        let window_scale = window.height().min(window.width());
        pitch -= (LOOK_SENSITIVITY * message.delta.y * window_scale).to_radians();
        yaw -= (LOOK_SENSITIVITY * message.delta.x * window_scale).to_radians();
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
        let forward = -transform.local_z().as_vec3();
        let right = transform.local_x().as_vec3();

        if let (Some(x), Some(y)) = (
            gamepad.get(GamepadAxis::LeftStickX),
            gamepad.get(GamepadAxis::LeftStickY),
        ) {
            if x.abs() > GAMEPAD_JOYSTICK_DEADZONE {
                velocity += right * x;
            }

            if y.abs() > GAMEPAD_JOYSTICK_DEADZONE {
                velocity += forward * y;
            }
        }

        if let (Some(right), Some(left)) = (
            gamepad.get(GamepadButton::RightTrigger2),
            gamepad.get(GamepadButton::LeftTrigger2),
        ) {
            if right.abs() > GAMEPAD_TRIGGER_DEADZONE {
                velocity += Vec3::Y * right;
            }

            if left.abs() > GAMEPAD_TRIGGER_DEADZONE {
                velocity -= Vec3::Y * left;
            }
        }

        transform.translation += velocity * time.delta_secs() * GAMEPAD_MOVE_SENSITIVITY;

        let (mut yaw, mut pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);

        if let (Some(x), Some(y)) = (
            gamepad.get(GamepadAxis::RightStickX),
            gamepad.get(GamepadAxis::RightStickY),
        ) {
            if x.abs() > GAMEPAD_JOYSTICK_DEADZONE {
                yaw -= (GAMEPAD_LOOK_SENSITIVITY * x * time.delta_secs()).to_radians();
            }

            if y.abs() > GAMEPAD_JOYSTICK_DEADZONE {
                pitch += (GAMEPAD_LOOK_SENSITIVITY * y * time.delta_secs()).to_radians();
            }
        }

        pitch = pitch.clamp(-1.54, 1.54);
        transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }
}

fn update_cursor_grab(
    mut cursor_options: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    let Ok(mut cursor_options) = cursor_options.single_mut() else {
        return;
    };

    if mouse.just_pressed(MouseButton::Left) {
        cursor_options.grab_mode = CursorGrabMode::Confined;
        cursor_options.visible = false;
    }

    if keys.just_pressed(KeyCode::Escape) {
        cursor_options.grab_mode = CursorGrabMode::None;
        cursor_options.visible = true;
    }
}
