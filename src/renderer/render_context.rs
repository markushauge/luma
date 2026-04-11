use anyhow::{Result, anyhow};
use ash::vk;

use super::{render_device::RenderDevice, render_queue::RenderQueue};

pub struct RenderContext {
    pub render_device: RenderDevice,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub semaphore: vk::Semaphore,
    pub fence: vk::Fence,
}

impl RenderContext {
    pub fn new(render_device: RenderDevice, render_queue: &RenderQueue) -> Result<Self> {
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

        Ok(Self {
            render_device,
            command_pool,
            command_buffer,
            semaphore,
            fence,
        })
    }
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe {
            self.render_device.device.destroy_fence(self.fence, None);

            self.render_device
                .device
                .destroy_semaphore(self.semaphore, None);

            self.render_device
                .device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);

            self.render_device
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
