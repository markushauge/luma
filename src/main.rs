mod asset;
mod renderer;

use bevy::{
    app::{App, AppExit},
    DefaultPlugins,
};

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(renderer::RendererPlugin)
        .run()
}
