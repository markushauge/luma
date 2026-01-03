mod compute_pipeline;
mod device;
mod frame;
mod schedule;
mod swapchain;

use std::time::Duration;

use anyhow::Result;
use ash::vk;
use bevy::{
    prelude::*,
    window::{PrimaryWindow, RawHandleWrapper},
};

use crate::{
    camera::Camera,
    shader::{Shader, ShaderPlugin},
};

use self::{
    compute_pipeline::ComputePipeline,
    device::Device,
    frame::Frame,
    schedule::{
        Render, RenderStartup, RenderSystems, run_render_schedule, run_render_startup_schedule,
    },
    swapchain::Swapchain,
};

#[derive(Resource, Clone)]
pub struct RendererSettings {
    pub resolution_scaling: f32,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            resolution_scaling: 1.0,
        }
    }
}

#[derive(Resource)]
struct ComputeShader(Handle<Shader>);

#[derive(States, Default, Debug, Hash, PartialEq, Eq, Clone)]
enum RendererState {
    #[default]
    Loading,
    Ready,
}

#[derive(Default)]
pub struct RendererPlugin {
    pub settings: RendererSettings,
}

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ShaderPlugin)
            .init_state::<RendererState>()
            .insert_resource(self.settings.clone())
            .add_schedule(RenderStartup::schedule())
            .add_schedule(Render::schedule())
            .add_systems(OnEnter(RendererState::Loading), load_shader)
            .add_systems(
                Update,
                check_shader_loaded.run_if(in_state(RendererState::Loading)),
            )
            .add_systems(
                OnEnter(RendererState::Ready),
                (setup_renderer, run_render_startup_schedule).chain(),
            )
            .add_systems(
                Update,
                run_render_schedule.run_if(in_state(RendererState::Ready)),
            )
            .add_systems(
                Render,
                (
                    begin.in_set(RenderSystems::Begin),
                    render.in_set(RenderSystems::Render),
                    end.in_set(RenderSystems::End),
                ),
            );
    }
}

fn load_shader(mut commands: Commands, asset_server: Res<AssetServer>) {
    let shader = asset_server.load("shaders/main.comp");
    commands.insert_resource(ComputeShader(shader));
}

fn check_shader_loaded(
    compute_shader: Res<ComputeShader>,
    assets: Res<Assets<Shader>>,
    mut next_state: ResMut<NextState<RendererState>>,
) {
    if assets.get(&compute_shader.0).is_some() {
        next_state.set(RendererState::Ready);
    }
}

fn setup_renderer(
    mut commands: Commands,
    windows: Query<(&Window, &RawHandleWrapper), With<PrimaryWindow>>,
    settings: Res<RendererSettings>,
    compute_shader: Res<ComputeShader>,
    assets: Res<Assets<Shader>>,
) -> Result<(), BevyError> {
    let shader = assets.get(&compute_shader.0).unwrap();
    let (window, raw_handles) = windows.single()?;
    let UVec2 { x, y } = window.physical_size();
    let renderer = Renderer::new(raw_handles, x, y, &settings, shader)?;
    commands.insert_resource(renderer);
    Ok(())
}

fn begin(mut renderer: ResMut<Renderer>) -> Result<(), BevyError> {
    renderer.begin().map_err(Into::into)
}

fn render(
    mut renderer: ResMut<Renderer>,
    query: Query<&Transform, With<Camera>>,
    time: Res<Time>,
) -> Result<(), BevyError> {
    let camera_transform = query.single()?;
    renderer.render(time.elapsed(), camera_transform)?;
    Ok(())
}

fn end(mut renderer: ResMut<Renderer>) -> Result<(), BevyError> {
    renderer.end().map_err(Into::into)
}

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

#[derive(Resource)]
pub struct Renderer {
    device: Device,
    swapchain: Swapchain,
    compute_pipeline: ComputePipeline,
    frames: Vec<Frame>,
    frame_index: usize,
}

impl Renderer {
    pub fn new(
        raw_handles: &RawHandleWrapper,
        width: u32,
        height: u32,
        settings: &RendererSettings,
        shader: &Shader,
    ) -> Result<Self> {
        let device = Device::new(raw_handles)?;
        let swapchain = Swapchain::new(device.clone(), raw_handles, width, height)?;
        let compute_pipeline = ComputePipeline::new(device.clone(), shader)?;

        let frame_width = (width as f32 * settings.resolution_scaling) as u32;
        let frame_height = (height as f32 * settings.resolution_scaling) as u32;

        let frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Frame::new(
                    &device,
                    frame_width,
                    frame_height,
                    &compute_pipeline.descriptor_set_layout_bindings,
                    compute_pipeline.descriptor_set_layout,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let frame_index = 0;

        Ok(Self {
            device,
            swapchain,
            compute_pipeline,
            frames,
            frame_index,
        })
    }

    pub fn begin(&mut self) -> Result<()> {
        let frame = &self.frames[self.frame_index];
        self.device.begin_frame(frame.command_buffer, frame.fence)?;
        self.swapchain.acquire_next(frame.semaphore)?;
        let present_image = self.swapchain.present_image();

        self.device.transition_image(
            frame.command_buffer,
            present_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        Ok(())
    }

    pub fn end(&mut self) -> Result<()> {
        let frame = &self.frames[self.frame_index];
        let present_image = self.swapchain.present_image();

        self.device.transition_image(
            frame.command_buffer,
            present_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        self.device.end_frame(
            frame.command_buffer,
            frame.semaphore,
            present_image.semaphore,
            frame.fence,
        )?;

        self.swapchain.present()?;
        self.frame_index = (self.frame_index + 1) % self.frames.len();
        Ok(())
    }

    pub fn render(&mut self, elapsed: Duration, camera_transform: &Transform) -> Result<()> {
        let frame = &self.frames[self.frame_index];
        let present_image = self.swapchain.present_image();
        let time_millis = elapsed.as_millis() as u32;

        self.compute_pipeline
            .dispatch(frame, camera_transform, time_millis);

        self.compute_pipeline
            .blit(frame, present_image.image, self.swapchain.surface_extent);

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.device_wait_idle().unwrap();
        }

        for frame in self.frames.drain(..) {
            frame.destroy(&self.device);
        }
    }
}
