mod camera;
mod panic;
mod renderer;

use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};

use crate::{
    camera::{Camera, CameraPlugin},
    renderer::{
        RendererPlugin,
        egui_renderer::{EguiContext, EguiPass, EguiPassSystems, EguiPlugin},
        ray_tracing::RayTracingPlugin,
    },
};

fn main() -> AppExit {
    panic::init_hook();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Luma".to_owned(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(RendererPlugin)
        .add_plugins(CameraPlugin)
        .add_plugins(RayTracingPlugin::default())
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(EguiPass, render_ui.in_set(EguiPassSystems::Render))
        .run()
}

fn setup(mut commands: Commands, mut assets: ResMut<Assets<Mesh>>) {
    commands.spawn((
        Camera::default(),
        Transform::from_xyz(0.0, 10.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let cube = assets.add(Cuboid::new(1.0, 1.0, 1.0).mesh().build());
    let sphere = assets.add(Sphere::new(0.5).mesh().build());

    for z in -5..=5 {
        let z = -z as f32 * 2.0;

        for x in -5..=5 {
            let x = x as f32 * 2.0;

            commands.spawn((Transform::from_xyz(x, 0.0, z), Mesh3d(cube.clone())));
            commands.spawn((Transform::from_xyz(x, 2.0, z), Mesh3d(sphere.clone())));
        }
    }
}

fn render_ui(
    ctx: Res<EguiContext>,
    diagnostics: Res<DiagnosticsStore>,
    mut camera: Query<&mut Camera>,
) {
    egui::Window::new("Stats")
        .collapsible(false)
        .resizable(false)
        .movable(false)
        .show(&ctx, |ui| {
            if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
                let fps = fps.smoothed().unwrap_or_default();
                ui.label(format!("FPS: {:.0}", fps));
            }

            if let Some(frame_time) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) {
                let frame_time = frame_time.smoothed().unwrap_or_default();
                ui.label(format!("Frame Time: {:.2} ms", frame_time));
            }
        });

    if let Ok(mut camera) = camera.single_mut() {
        egui::Window::new("Camera")
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .show(&ctx, |ui| {
                ui.label("Focal length:");
                ui.add(
                    egui::Slider::new(&mut camera.focal_length, 10.0..=70.0)
                        .step_by(1.0)
                        .custom_formatter(|value, _| format!("{:.0} mm", value)),
                );
            });
    }
}
