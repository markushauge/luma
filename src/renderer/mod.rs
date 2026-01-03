pub mod compute_pipeline;
mod device;
mod schedule;
mod swapchain;

use anyhow::{Result, anyhow};
use ash::vk;
use bevy::{
    prelude::*,
    window::{PrimaryWindow, RawHandleWrapper},
};

use crate::shader::ShaderPlugin;

use self::{
    device::Device,
    schedule::{
        Render, RenderStartup, RenderSystems, run_render_schedule, run_render_startup_schedule,
    },
    swapchain::Swapchain,
};

pub struct RendererPlugin;

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ShaderPlugin)
            .add_schedule(RenderStartup::schedule())
            .add_schedule(Render::schedule())
            .add_systems(
                Startup,
                (setup_renderer, run_render_startup_schedule).chain(),
            )
            .add_systems(Update, run_render_schedule)
            .add_systems(
                Render,
                (
                    begin.in_set(RenderSystems::Begin),
                    end.in_set(RenderSystems::End),
                ),
            );
    }
}

fn setup_renderer(
    mut commands: Commands,
    windows: Query<(&Window, &RawHandleWrapper), With<PrimaryWindow>>,
) -> Result<(), BevyError> {
    let (window, raw_handles) = windows.single()?;
    let UVec2 { x, y } = window.physical_size();
    let renderer = Renderer::new(raw_handles, x, y)?;
    commands.insert_resource(renderer);
    Ok(())
}

fn begin(mut renderer: ResMut<Renderer>) -> Result<(), BevyError> {
    renderer.begin().map_err(Into::into)
}

fn end(mut renderer: ResMut<Renderer>) -> Result<(), BevyError> {
    renderer.end().map_err(Into::into)
}

#[derive(Resource)]
#[allow(dead_code)]
pub struct Renderer {
    device: Device,
    swapchain: Swapchain,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl Renderer {
    pub fn new(raw_handles: &RawHandleWrapper, width: u32, height: u32) -> Result<Self> {
        let device = Device::new(raw_handles)?;
        let swapchain = Swapchain::new(device.clone(), raw_handles, width, height)?;

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
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.device_wait_idle().unwrap();
        }
    }
}
