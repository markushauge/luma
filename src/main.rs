mod camera;
mod panic;
mod renderer;
mod shader;

use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    window::PrimaryWindow,
};

use crate::{
    camera::{Camera, CameraPlugin},
    renderer::{RendererPlugin, RendererSettings},
};

fn main() -> AppExit {
    panic::init_hook();

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(RendererPlugin {
            settings: RendererSettings {
                resolution_scaling: 0.25,
            },
        })
        .add_plugins(CameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, update_window_title)
        .run()
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera,
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn update_window_title(
    mut window: Query<&mut Window, With<PrimaryWindow>>,
    mut timer: Local<Option<Timer>>,
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time>,
) {
    let Ok(mut window) = window.single_mut() else {
        return;
    };

    match timer.as_mut() {
        Some(timer) => {
            if !timer.tick(time.delta()).is_finished() {
                return;
            }
        }
        None => {
            *timer = Some(Timer::from_seconds(1.0, TimerMode::Repeating));
        }
    }

    let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) else {
        return;
    };

    let Some(frame_time) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) else {
        return;
    };

    window.title = format!(
        "Luma ({:.0} FPS, {:.2} ms)",
        fps.smoothed().unwrap_or_default(),
        frame_time.smoothed().unwrap_or_default()
    );
}
