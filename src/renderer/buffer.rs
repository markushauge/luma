use std::marker::PhantomData;

use anyhow::{Result, anyhow};
use ash::vk;
use bevy::prelude::*;
use bytemuck::Pod;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use super::render_device::RenderDevice;

#[expect(dead_code)]
pub struct Buffer<T = u8> {
    pub len: u64,
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub address: vk::DeviceAddress,
    pub name: String,
    _marker: PhantomData<T>,
}

impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self {
            len: default(),
            size: default(),
            usage: default(),
            buffer: default(),
            allocation: default(),
            address: default(),
            name: default(),
            _marker: default(),
        }
    }
}

impl RenderDevice {
    pub fn create_buffer<T>(
        &self,
        len: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: Option<&str>,
    ) -> Result<Buffer<T>, vk::Result> {
        unsafe {
            let size = len * size_of::<T>() as u64;

            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = self.device.create_buffer(&buffer_create_info, None)?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let name = name.unwrap_or("Buffer").to_string();

            let allocation = self.allocate(&AllocationCreateDesc {
                name: &name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
            });

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;

            let address = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));

            Ok(Buffer {
                len,
                size,
                usage,
                buffer,
                allocation,
                address,
                name,
                _marker: PhantomData,
            })
        }
    }

    pub fn destroy_buffer<T>(&self, buffer: Buffer<T>) {
        unsafe {
            self.device.destroy_buffer(buffer.buffer, None);
            self.free(buffer.allocation);
        }
    }
}

impl<T: Pod> Buffer<T> {
    pub fn slice_mut(&mut self) -> Result<&mut [T]> {
        self.allocation
            .mapped_slice_mut()
            .ok_or_else(|| anyhow!("Buffer is not host visible"))
            .map(|slice| &mut slice[..self.size as usize])
            .map(bytemuck::cast_slice_mut)
    }
}
