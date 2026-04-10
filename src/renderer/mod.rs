pub mod acceleration_structure;
mod buffer;
pub mod egui_renderer;
pub mod ray_tracing;
mod render_device;
mod render_queue;
mod resource_state_tracker;
mod schedule;
mod storage_image;
mod swapchain;

use anyhow::{Result, anyhow};
use ash::vk;
use bevy::{
    prelude::*,
    window::{PrimaryWindow, RawHandleWrapper},
};
use render_device::RenderDevice;
use render_queue::RenderQueue;
use resource_state_tracker::{ImageState, ResourceStateTracker};
use schedule::{Render, RenderStartup, run_render_startup_schedule};
use swapchain::Swapchain;

use crate::shader::ShaderPlugin;

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
    render_device: RenderDevice,
    render_queue: RenderQueue,
    swapchain: Swapchain,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    tracker: ResourceStateTracker,
}

impl Renderer {
    pub fn new(handle: RawHandleWrapper, width: u32, height: u32) -> Result<Self> {
        let (render_device, render_queue) = RenderDevice::new(&handle)?;

        let swapchain = Swapchain::new(
            render_device.clone(),
            &render_queue,
            &handle,
            width,
            height,
            None,
        )?;

        let command_pool = unsafe {
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(render_queue.queue_family_index);

            render_device
                .device
                .create_command_pool(&command_pool_create_info, None)?
        };

        let [command_buffer] = unsafe {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            render_device
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)?
                .try_into()
                .map_err(|_| anyhow!("Failed to allocate exactly one command buffer"))?
        };

        let semaphore = unsafe {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            render_device
                .device
                .create_semaphore(&semaphore_create_info, None)?
        };

        let fence = unsafe {
            let fence_create_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

            render_device
                .device
                .create_fence(&fence_create_info, None)?
        };

        let tracker = ResourceStateTracker::new();

        Ok(Self {
            handle,
            render_device,
            render_queue,
            swapchain,
            command_pool,
            command_buffer,
            semaphore,
            fence,
            tracker,
        })
    }

    pub fn begin(&mut self) -> Result<()> {
        self.render_device
            .begin_frame(self.command_buffer, self.fence)?;

        self.swapchain.acquire_next(self.semaphore)?;

        if self.swapchain.out_of_date {
            return Ok(());
        }

        let present_image = self.swapchain.present_image();

        self.tracker
            .track_image(present_image.image)
            .transition_image(
                present_image.image,
                ImageState {
                    layout: vk::ImageLayout::GENERAL,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                    stages: vk::PipelineStageFlags2::TRANSFER,
                },
            )
            .flush(&self.render_device, self.command_buffer);

        Ok(())
    }

    pub fn end(&mut self) -> Result<()> {
        let present_image = self.swapchain.present_image();

        self.tracker
            .transition_image(
                present_image.image,
                ImageState {
                    layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    access: vk::AccessFlags2::empty(),
                    stages: vk::PipelineStageFlags2::empty(),
                },
            )
            .flush(&self.render_device, self.command_buffer);

        self.render_device.end_frame(
            &self.render_queue,
            self.command_buffer,
            self.semaphore,
            present_image.semaphore,
            self.fence,
        )?;

        self.swapchain.present(&self.render_queue)?;

        Ok(())
    }

    pub fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        self.render_device.wait_idle()?;

        for present_image in &self.swapchain.present_images {
            self.tracker.untrack_image(present_image.image);
        }

        self.swapchain = Swapchain::new(
            self.render_device.clone(),
            &self.render_queue,
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
        self.render_device.wait_idle().unwrap();
    }
}
