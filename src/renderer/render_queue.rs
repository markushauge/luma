use ash::vk;
use bevy::prelude::*;

use super::render_device::RenderDevice;

#[expect(dead_code)]
pub struct RenderQueue {
    pub render_device: RenderDevice,
    pub queue_family_index: u32,
    pub queue_index: u32,
    pub queue: vk::Queue,
}

impl RenderQueue {
    pub fn new(render_device: RenderDevice, queue_family_index: u32, queue_index: u32) -> Self {
        let queue = unsafe {
            render_device
                .device
                .get_device_queue(queue_family_index, queue_index)
        };

        Self {
            render_device,
            queue_family_index,
            queue_index,
            queue,
        }
    }

    pub unsafe fn submit(
        &self,
        command_buffer: vk::CommandBuffer,
        wait_semaphore: vk::Semaphore,
        signal_semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> Result<(), vk::Result> {
        unsafe {
            let submit_info = vk::SubmitInfo::default()
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])
                .command_buffers(std::slice::from_ref(&command_buffer))
                .wait_semaphores(std::slice::from_ref(&wait_semaphore))
                .signal_semaphores(std::slice::from_ref(&signal_semaphore));

            self.render_device
                .device
                .queue_submit(self.queue, &[submit_info], fence)
        }
    }
}
