pub mod compute_pipeline;
mod device;
pub mod egui_renderer;
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
    schedule::{Render, RenderStartup, run_render_startup_schedule},
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
            .add_systems(Update, (recreate_swapchain, render).chain());
    }
}

fn setup_renderer(
    mut commands: Commands,
    windows: Query<(&Window, &RawHandleWrapper), With<PrimaryWindow>>,
) -> Result<(), BevyError> {
    let (window, handle) = windows.single()?;
    let width = window.physical_width();
    let height = window.physical_height();
    let renderer = Renderer::new(handle.clone(), width, height)?;
    commands.insert_resource(renderer);
    Ok(())
}

fn recreate_swapchain(
    mut renderer: ResMut<Renderer>,
    windows: Query<&Window, With<PrimaryWindow>>,
) -> Result<(), BevyError> {
    if !renderer.swapchain.out_of_date {
        return Ok(());
    }

    let window = windows.single()?;
    let width = window.physical_width();
    let height = window.physical_height();
    renderer.recreate_swapchain(width, height)?;
    Ok(())
}

fn render(world: &mut World) -> Result<(), BevyError> {
    let mut renderer = world.resource_mut::<Renderer>();

    if !renderer.swapchain.out_of_date {
        renderer.begin()?;
    }

    if !renderer.swapchain.out_of_date {
        world.run_schedule(Render);
    }

    let mut renderer = world.resource_mut::<Renderer>();

    if !renderer.swapchain.out_of_date {
        renderer.end()?;
    }

    Ok(())
}

#[derive(Resource)]
pub struct Renderer {
    handle: RawHandleWrapper,
    device: Device,
    swapchain: Swapchain,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl Renderer {
    pub fn new(handle: RawHandleWrapper, width: u32, height: u32) -> Result<Self> {
        let device = Device::new(&handle)?;
        let swapchain = Swapchain::new(device.clone(), &handle, width, height, None)?;

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
            handle,
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

        if self.swapchain.out_of_date {
            return Ok(());
        }

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

    pub fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        self.device.wait_idle()?;

        self.swapchain = Swapchain::new(
            self.device.clone(),
            &self.handle,
            width,
            height,
            Some(&mut self.swapchain),
        )?;

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
    }
}
