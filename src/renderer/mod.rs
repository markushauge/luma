pub mod acceleration_structure;
mod buffer;
pub mod egui_renderer;
pub mod ray_tracing;
mod render_context;
mod render_device;
mod render_queue;
mod resource_state_tracker;
mod schedule;
mod shader;
mod storage_image;
mod swapchain;

use anyhow::Result;
use bevy::{
    ecs::system::SystemId,
    prelude::*,
    window::{PrimaryWindow, RawHandleWrapper},
};
use render_context::RenderContext;
use render_device::RenderDevice;
use render_queue::RenderQueue;
use resource_state_tracker::{ImageState, ResourceStateTracker};
use schedule::{Render, RenderStartup, run_render_startup_schedule};
use shader::ShaderPlugin;
use swapchain::Swapchain;

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
    let (render_device, render_queue) = RenderDevice::new(handle.get_display_handle())?;
    let render_context = RenderContext::new(render_device.clone(), &render_queue)?;

    let swapchain = Swapchain::new(
        render_device.clone(),
        &render_queue,
        handle.get_display_handle(),
        handle.get_window_handle(),
        window.physical_width(),
        window.physical_height(),
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

    render_device.wait_idle();

    for swapchain_image in &swapchain.swapchain_images {
        resource_state_tracker.untrack_image(swapchain_image.image);
    }

    let (window, handle) = windows.single()?;

    *swapchain = Swapchain::new(
        render_device.clone(),
        &render_queue,
        handle.get_display_handle(),
        handle.get_window_handle(),
        window.physical_width(),
        window.physical_height(),
        Some(&mut swapchain),
    )?;

    Ok(())
}

#[derive(Default, Deref, DerefMut)]
struct BeginSystemId(Option<SystemId<(), Result<bool>>>);

#[derive(Default, Deref, DerefMut)]
struct EndSystemId(Option<SystemId<(), Result<()>>>);

fn render(
    world: &mut World,
    mut begin_system_id: Local<BeginSystemId>,
    mut end_system_id: Local<EndSystemId>,
) -> Result<(), BevyError> {
    let begin_system_id = *begin_system_id.get_or_insert_with(|| world.register_system(begin));
    let end_system_id = *end_system_id.get_or_insert_with(|| world.register_system(end));
    let out_of_date = world.run_system(begin_system_id)??;

    if out_of_date {
        return Ok(());
    }

    world.run_schedule(Render);
    world.run_system(end_system_id)??;
    Ok(())
}

fn begin(
    render_device: Res<RenderDevice>,
    render_context: Res<RenderContext>,
    mut swapchain: ResMut<Swapchain>,
    mut resource_state_tracker: ResMut<ResourceStateTracker>,
) -> Result<bool> {
    render_context.begin_frame(&render_device)?;
    swapchain.acquire_next(render_context.semaphore)?;

    if swapchain.out_of_date {
        return Ok(true);
    }

    let swapchain_image = swapchain.current_image();
    resource_state_tracker.track_image(swapchain_image.image);
    Ok(false)
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
        .transition_image(swapchain_image.image, ImageState::present())
        .flush(&render_device, render_context.command_buffer);

    render_context.end_frame(&render_device, &render_queue, swapchain_image.semaphore)?;
    swapchain.present(&render_queue)?;
    Ok(())
}

fn on_app_exit(mut app_exit_events: MessageReader<AppExit>, render_device: Res<RenderDevice>) {
    if !app_exit_events.is_empty() {
        app_exit_events.clear();
        render_device.wait_idle();
    }
}
