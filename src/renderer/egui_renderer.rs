use anyhow::Result;
use ash::vk;
use bevy::prelude::*;
use egui::{Context, FullOutput, RawInput, TextureId};
use egui_ash_renderer::{DynamicRendering, Options, Renderer};

use super::Device;

pub struct EguiRenderer {
    pub device: Device,
    pub context: Context,
    pub renderer: Renderer,
    pub textures_to_free: Vec<TextureId>,
}

impl EguiRenderer {
    pub fn new(device: Device, in_flight_frames: usize) -> Result<Self> {
        let context = Context::default();

        let dynamic_rendering = DynamicRendering {
            color_attachment_format: vk::Format::B8G8R8A8_UNORM,
            depth_attachment_format: None,
        };

        let options = Options {
            in_flight_frames,
            srgb_framebuffer: true,
            ..default()
        };

        let renderer = Renderer::with_gpu_allocator(
            device.allocator.clone(),
            device.device.clone(),
            dynamic_rendering,
            options,
        )?;

        let textures_to_free = Vec::new();

        Ok(Self {
            device,
            context,
            renderer,
            textures_to_free,
        })
    }

    pub fn render(
        &mut self,
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        image_view: vk::ImageView,
        run_ui: impl FnMut(&Context),
    ) -> Result<()> {
        if !self.textures_to_free.is_empty() {
            self.renderer.free_textures(&self.textures_to_free)?;
            self.textures_to_free.clear();
        }

        // TODO: Get raw input from egui-winit
        let raw_input = RawInput::default();

        let FullOutput {
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = self.context.run(raw_input, run_ui);

        if !textures_delta.free.is_empty() {
            self.textures_to_free.extend(textures_delta.free);
        }

        if !textures_delta.set.is_empty() {
            self.renderer
                .set_textures(self.device.queue, command_pool, &textures_delta.set)?;
        }

        let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);

        let rendering_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&rendering_attachment_info));

        unsafe {
            self.device
                .device
                .cmd_begin_rendering(command_buffer, &rendering_info);

            self.renderer.cmd_draw(
                command_buffer,
                extent,
                pixels_per_point,
                &clipped_primitives,
            )?;

            self.device.device.cmd_end_rendering(command_buffer);
        }

        Ok(())
    }
}
