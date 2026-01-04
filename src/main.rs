mod camera;
mod panic;
mod renderer;
mod shader;

use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};

use crate::{
    camera::{Camera, CameraPlugin},
    renderer::{
        RendererPlugin,
        compute_pipeline::{ComputePipelinePlugin, ComputePipelineSettings},
        egui_renderer::{EguiContext, EguiPass, EguiPassSystems, EguiPlugin},
    },
};

fn main() -> AppExit {
    panic::init_hook();

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(RendererPlugin)
        .add_plugins(CameraPlugin)
        .add_plugins(ComputePipelinePlugin {
            settings: ComputePipelineSettings {
                resolution_scaling: 0.25,
            },
        })
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(EguiPass, render_ui.in_set(EguiPassSystems::Render))
        .run()
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera,
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn render_ui(ctx: Res<EguiContext>, diagnostics: Res<DiagnosticsStore>) {
    egui::Window::new("Stats").show(&ctx, |ui| {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            let fps = fps.smoothed().unwrap_or_default();
            ui.label(format!("FPS: {:.0}", fps));
        }

        if let Some(frame_time) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) {
            let frame_time = frame_time.smoothed().unwrap_or_default();
            ui.label(format!("Frame Time: {:.2} ms", frame_time));
        }
    });
}
