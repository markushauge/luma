mod renderer;
mod shader;

use bevy::{
    app::{App, AppExit},
    DefaultPlugins,
};

use crate::renderer::RendererPlugin;

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RendererPlugin)
        .run()
}
