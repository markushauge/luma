use anyhow::{Result, anyhow};
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use super::render_device::RenderDevice;

#[derive(Default)]
#[expect(dead_code)]
pub struct Buffer {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
}

impl RenderDevice {
    pub fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: Option<&str>,
    ) -> Result<Buffer, vk::Result> {
        unsafe {
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = self.device.create_buffer(&buffer_create_info, None)?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let allocation = self.allocate(&AllocationCreateDesc {
                name: name.unwrap_or("Buffer"),
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
            });

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;

            Ok(Buffer {
                size,
                usage,
                buffer,
                allocation,
            })
        }
    }

    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe {
            self.device.destroy_buffer(buffer.buffer, None);
            self.free(buffer.allocation);
        }
    }

    pub fn get_buffer_device_address(&self, buffer: &Buffer) -> vk::DeviceAddress {
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer),
            )
        }
    }
}

impl Buffer {
    pub fn slice_mut(&mut self) -> Result<&mut [u8]> {
        self.allocation
            .mapped_slice_mut()
            .ok_or_else(|| anyhow!("Buffer is not host visible"))
            .map(|slice| &mut slice[..self.size as usize])
    }
}
