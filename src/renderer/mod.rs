pub mod acceleration_structure;
mod buffer;
pub mod egui_renderer;
pub mod ray_tracing;
mod render_context;
mod render_device;
mod render_queue;
mod resource_state_tracker;
mod schedule;
mod storage_image;
mod swapchain;

use anyhow::Result;
use ash::vk;
use bevy::{
    ecs::system::{RunSystemError, RunSystemOnce},
    prelude::*,
    window::{PrimaryWindow, RawHandleWrapper},
};
use render_context::RenderContext;
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
            .add_systems(Update, (recreate_swapchain, render).chain())
            .add_systems(Last, on_app_exit);
    }
}

fn setup_renderer(
    mut commands: Commands,
    windows: Query<(&Window, &RawHandleWrapper), With<PrimaryWindow>>,
) -> Result<(), BevyError> {
    let (window, handle) = windows.single()?;
    let width = window.physical_width();
    let height = window.physical_height();
    let (render_device, render_queue) = RenderDevice::new(&handle)?;
    let render_context = RenderContext::new(render_device.clone(), &render_queue)?;

    let swapchain = Swapchain::new(
        render_device.clone(),
        &render_queue,
        &handle,
        width,
        height,
        None,
    )?;

    commands.insert_resource(render_device);
    commands.insert_resource(render_queue);
    commands.insert_resource(render_context);
    commands.insert_resource(swapchain);
    commands.init_resource::<ResourceStateTracker>();

    Ok(())
}

fn recreate_swapchain(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut swapchain: ResMut<Swapchain>,
    mut resource_state_tracker: ResMut<ResourceStateTracker>,
    windows: Query<(&Window, &RawHandleWrapper), With<PrimaryWindow>>,
) -> Result<(), BevyError> {
    if !swapchain.out_of_date {
        return Ok(());
    }

    let (window, handle) = windows.single()?;
    let width = window.physical_width();
    let height = window.physical_height();

    render_device.wait_idle();

    for swapchain_image in &swapchain.swapchain_images {
        resource_state_tracker.untrack_image(swapchain_image.image);
    }

    *swapchain = Swapchain::new(
        render_device.clone(),
        &render_queue,
        &handle,
        width,
        height,
        Some(&mut swapchain),
    )?;

    Ok(())
}

fn render(world: &mut World) -> Result<(), BevyError> {
    if !is_swapchain_out_of_date(world) {
        if let Err(RunSystemError::Failed(err)) = world.run_system_once(begin) {
            return Err(err);
        };
    }

    if !is_swapchain_out_of_date(world) {
        world.run_schedule(Render);
    }

    if !is_swapchain_out_of_date(world) {
        if let Err(RunSystemError::Failed(err)) = world.run_system_once(end) {
            return Err(err);
        };
    }

    Ok(())
}

fn is_swapchain_out_of_date(world: &World) -> bool {
    world.resource::<Swapchain>().out_of_date
}

fn begin(
    render_device: Res<RenderDevice>,
    render_context: Res<RenderContext>,
    mut swapchain: ResMut<Swapchain>,
    mut resource_state_tracker: ResMut<ResourceStateTracker>,
) -> Result<()> {
    render_device.begin_frame(render_context.command_buffer, render_context.fence)?;
    swapchain.acquire_next(render_context.semaphore)?;

    if swapchain.out_of_date {
        return Ok(());
    }

    let swapchain_image = swapchain.current_image();

    resource_state_tracker
        .track_image(swapchain_image.image)
        .transition_image(
            swapchain_image.image,
            ImageState {
                layout: vk::ImageLayout::GENERAL,
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stages: vk::PipelineStageFlags2::TRANSFER,
            },
        )
        .flush(&render_device, render_context.command_buffer);

    Ok(())
}

fn end(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    render_context: Res<RenderContext>,
    mut swapchain: ResMut<Swapchain>,
    mut resource_state_tracker: ResMut<ResourceStateTracker>,
) -> Result<()> {
    let swapchain_image = swapchain.current_image();

    resource_state_tracker
        .transition_image(
            swapchain_image.image,
            ImageState {
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                access: vk::AccessFlags2::empty(),
                stages: vk::PipelineStageFlags2::empty(),
            },
        )
        .flush(&render_device, render_context.command_buffer);

    render_device.end_frame(
        &render_queue,
        render_context.command_buffer,
        render_context.semaphore,
        swapchain_image.semaphore,
        render_context.fence,
    )?;

    swapchain.present(&render_queue)?;
    Ok(())
}

fn on_app_exit(mut app_exit_events: MessageReader<AppExit>, render_device: Res<RenderDevice>) {
    if let Some(_) = app_exit_events.read().next() {
        render_device.wait_idle();
    }
}
