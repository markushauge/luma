use std::collections::HashMap;

use ash::vk;

use super::device::Device;

pub struct ResourceStateTracker {
    images: HashMap<vk::Image, ImageState>,
    image_barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
}

impl ResourceStateTracker {
    pub fn new() -> Self {
        Self {
            images: HashMap::new(),
            image_barriers: Vec::new(),
        }
    }

    pub fn track_image(&mut self, image: vk::Image) -> &mut Self {
        self.images.insert(image, ImageState::default());
        self
    }

    pub fn untrack_image(&mut self, image: vk::Image) -> &mut Self {
        self.images.remove(&image);
        self
    }

    pub fn transition_image(&mut self, image: vk::Image, state_after: ImageState) -> &mut Self {
        let state_before = self.images.entry(image).or_default();

        if *state_before == state_after {
            return self;
        }

        self.image_barriers.push(
            vk::ImageMemoryBarrier2::default()
                .old_layout(state_before.layout)
                .new_layout(state_after.layout)
                .src_access_mask(state_before.access)
                .dst_access_mask(state_after.access)
                .src_stage_mask(state_before.stages)
                .dst_stage_mask(state_after.stages)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        );

        *state_before = state_after;
        self
    }

    pub fn flush(&mut self, device: &Device, command_buffer: vk::CommandBuffer) -> &mut Self {
        if self.image_barriers.is_empty() {
            return self;
        }

        unsafe {
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(&self.image_barriers);

            device
                .device
                .cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }

        self.image_barriers.clear();
        self
    }
}

#[derive(Clone, PartialEq)]
pub struct ImageState {
    pub layout: vk::ImageLayout,
    pub access: vk::AccessFlags2,
    pub stages: vk::PipelineStageFlags2,
}

impl Default for ImageState {
    fn default() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            access: vk::AccessFlags2::empty(),
            stages: vk::PipelineStageFlags2::empty(),
        }
    }
}
