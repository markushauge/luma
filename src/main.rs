mod camera;
mod renderer;
mod shader;

use bevy::prelude::*;

use crate::{
    camera::{Camera, CameraPlugin},
    renderer::RendererPlugin,
};

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RendererPlugin)
        .add_plugins(CameraPlugin)
        .add_systems(Startup, setup)
        .run()
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera,
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
