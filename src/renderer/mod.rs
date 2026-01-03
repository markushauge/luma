mod compute_pipeline;
mod device;
mod schedule;
mod swapchain;

use std::time::Duration;

use anyhow::{Result, anyhow};
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

#[derive(Resource)]
#[allow(dead_code)]
pub struct Renderer {
    device: Device,
    swapchain: Swapchain,
    compute_pipeline: ComputePipeline,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
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

        let frame_width = (width as f32 * settings.resolution_scaling) as u32;
        let frame_height = (height as f32 * settings.resolution_scaling) as u32;

        let compute_pipeline =
            ComputePipeline::new(device.clone(), shader, frame_width, frame_height)?;

        let command_pool = unsafe {
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(device.queue_family_index);

            device
                .device
                .create_command_pool(&command_pool_create_info, None)?
        };

        let [command_buffer] = unsafe {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            device
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)?
                .try_into()
                .map_err(|_| anyhow!("Failed to allocate exactly one command buffer"))?
        };

        let semaphore = unsafe {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            device
                .device
                .create_semaphore(&semaphore_create_info, None)?
        };

        let fence = unsafe {
            let fence_create_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

            device.device.create_fence(&fence_create_info, None)?
        };

        Ok(Self {
            device,
            swapchain,
            compute_pipeline,
            command_pool,
            command_buffer,
            semaphore,
            fence,
        })
    }

    pub fn begin(&mut self) -> Result<()> {
        self.device.begin_frame(self.command_buffer, self.fence)?;
        self.swapchain.acquire_next(self.semaphore)?;
        let present_image = self.swapchain.present_image();

        self.device.transition_image(
            self.command_buffer,
            present_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        Ok(())
    }

    pub fn end(&mut self) -> Result<()> {
        let present_image = self.swapchain.present_image();

        self.device.transition_image(
            self.command_buffer,
            present_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        self.device.end_frame(
            self.command_buffer,
            self.semaphore,
            present_image.semaphore,
            self.fence,
        )?;

        self.swapchain.present()?;
        Ok(())
    }

    pub fn render(&mut self, elapsed: Duration, camera_transform: &Transform) -> Result<()> {
        let present_image = self.swapchain.present_image();
        let time_millis = elapsed.as_millis() as u32;

        self.compute_pipeline
            .dispatch(self.command_buffer, camera_transform, time_millis);

        self.compute_pipeline.blit(
            self.command_buffer,
            present_image.image,
            self.swapchain.surface_extent,
        );

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.device_wait_idle().unwrap();
        }
    }
}
