use anyhow::Result;
use ash::vk;
use bevy::{
    ecs::schedule::ScheduleLabel,
    prelude::*,
    window::PrimaryWindow,
    winit::{DisplayHandleWrapper, RawWinitWindowEvent, WINIT_WINDOWS},
};
use egui::ViewportId;

use super::{
    render_context::RenderContext,
    render_device::RenderDevice,
    render_queue::RenderQueue,
    resource_state_tracker::{ImageState, ResourceStateTracker},
    schedule::{Render, RenderStartup, RenderSystems},
    swapchain::Swapchain,
};

pub struct EguiPlugin;

impl Plugin for EguiPlugin {
    fn build(&self, app: &mut App) {
        app.add_schedule(EguiPass::schedule())
            .add_systems(RenderStartup, setup)
            .add_systems(Render, run_egui_pass.in_set(RenderSystems::QueueUi))
            .add_systems(
                EguiPass,
                (
                    begin.in_set(EguiPassSystems::Begin),
                    end.in_set(EguiPassSystems::End),
                ),
            );
    }
}

#[derive(ScheduleLabel, Debug, Hash, PartialEq, Eq, Clone)]
pub struct EguiPass;

impl EguiPass {
    pub fn schedule() -> Schedule {
        let mut schedule = Schedule::new(Self);

        schedule.configure_sets(
            (
                EguiPassSystems::Begin,
                EguiPassSystems::Render,
                EguiPassSystems::End,
            )
                .chain(),
        );

        schedule
    }
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum EguiPassSystems {
    Begin,
    Render,
    End,
}

#[derive(Resource, Deref, DerefMut)]
pub struct EguiState(egui_winit::State);

#[derive(Resource, Deref)]
pub struct EguiContext(egui::Context);

fn setup(
    mut commands: Commands,
    display_handle: Res<DisplayHandleWrapper>,
    render_device: Res<RenderDevice>,
) -> Result<(), BevyError> {
    let context = egui::Context::default();

    let state = egui_winit::State::new(
        context.clone(),
        ViewportId::ROOT,
        &**display_handle,
        None,
        None,
        None,
    );

    let egui_renderer = EguiRenderer::new(render_device.clone(), context.clone())?;
    commands.insert_resource(EguiContext(context));
    commands.insert_resource(EguiState(state));
    commands.insert_resource(egui_renderer);
    Ok(())
}

fn run_egui_pass(world: &mut World) {
    world.run_schedule(EguiPass);
}

fn begin(
    mut egui_renderer: ResMut<EguiRenderer>,
    mut state: ResMut<EguiState>,
    windows: Query<Entity, With<PrimaryWindow>>,
    mut winit_events: MessageReader<RawWinitWindowEvent>,
) -> Result<(), BevyError> {
    let window = windows.single()?;

    let raw_input = WINIT_WINDOWS.with_borrow(|windows| {
        let window = windows.get_window(window).unwrap();

        for event in winit_events.read() {
            if event.window_id == window.id() {
                let _ = state.on_window_event(window, &event.event);
            }
        }

        state.take_egui_input(window)
    });

    egui_renderer.begin(raw_input)?;
    Ok(())
}

fn end(
    render_queue: Res<RenderQueue>,
    render_context: Res<RenderContext>,
    swapchain: Res<Swapchain>,
    mut tracker: ResMut<ResourceStateTracker>,
    mut egui_renderer: ResMut<EguiRenderer>,
) -> Result<(), BevyError> {
    egui_renderer.end(&render_queue, &render_context, &swapchain, &mut tracker)?;
    Ok(())
}

#[derive(Resource)]
pub struct EguiRenderer {
    pub render_device: RenderDevice,
    pub context: egui::Context,
    pub renderer: egui_ash_renderer::Renderer,
    pub textures_to_free: Vec<egui::TextureId>,
}

impl EguiRenderer {
    pub fn new(render_device: RenderDevice, context: egui::Context) -> Result<Self> {
        let dynamic_rendering = egui_ash_renderer::DynamicRendering {
            color_attachment_format: vk::Format::B8G8R8A8_UNORM,
            depth_attachment_format: None,
        };

        let renderer = egui_ash_renderer::Renderer::with_gpu_allocator(
            render_device.allocator.clone(),
            render_device.device.clone(),
            dynamic_rendering,
            default(),
        )?;

        let textures_to_free = Vec::new();

        Ok(Self {
            render_device,
            context,
            renderer,
            textures_to_free,
        })
    }

    pub fn begin(&mut self, raw_input: egui::RawInput) -> Result<()> {
        if !self.textures_to_free.is_empty() {
            self.renderer.free_textures(&self.textures_to_free)?;
            self.textures_to_free.clear();
        }

        self.context.begin_pass(raw_input);
        Ok(())
    }

    pub fn end(
        &mut self,
        render_queue: &RenderQueue,
        render_context: &RenderContext,
        swapchain: &Swapchain,
        tracker: &mut ResourceStateTracker,
    ) -> Result<()> {
        let egui::FullOutput {
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = self.context.end_pass();

        if !textures_delta.free.is_empty() {
            self.textures_to_free.extend(textures_delta.free);
        }

        if !textures_delta.set.is_empty() {
            self.renderer.set_textures(
                render_queue.queue,
                render_context.command_pool,
                &textures_delta.set,
            )?;
        }

        let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
        let swapchain_image = swapchain.current_image();

        tracker
            .transition_image(swapchain_image.image, ImageState::color_attachment())
            .flush(&self.render_device, render_context.command_buffer);

        let rendering_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(swapchain_image.image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.surface_extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&rendering_attachment_info));

        unsafe {
            self.render_device
                .device
                .cmd_begin_rendering(render_context.command_buffer, &rendering_info);

            self.renderer.cmd_draw(
                render_context.command_buffer,
                swapchain.surface_extent,
                pixels_per_point,
                &clipped_primitives,
            )?;

            self.render_device
                .device
                .cmd_end_rendering(render_context.command_buffer);
        }

        Ok(())
    }
}
