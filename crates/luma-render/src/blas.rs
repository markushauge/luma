use anyhow::{Result, anyhow};
use ash::vk;
use gpu_allocator::MemoryLocation;

use super::{buffer::Buffer, mesh::Vertex, render_device::RenderDevice, render_queue::RenderQueue};

#[derive(Default)]
pub struct Blas {
    pub buffer: Buffer,
    pub acceleration_structure: vk::AccelerationStructureKHR,
    pub device_address: vk::DeviceAddress,
}

impl RenderDevice {
    pub fn create_blas(
        &self,
        render_queue: &RenderQueue,
        vertex_buffer: &Buffer<Vertex>,
        index_buffer: &Buffer<u32>,
    ) -> Result<Blas> {
        unsafe {
            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: vertex_buffer.address,
                        })
                        .vertex_stride(size_of::<Vertex>() as u64)
                        .max_vertex(vertex_buffer.len as u32 - 1)
                        .index_type(vk::IndexType::UINT32)
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: index_buffer.address,
                        }),
                });

            let primitive_count = (index_buffer.len / 3) as u32;

            let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(&geometry));

            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            self.acceleration_structure_device
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[primitive_count],
                    &mut size_info,
                );

            let buffer = self.create_buffer(
                size_info.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
                Some("BLAS Buffer"),
            )?;

            let acceleration_structure = self
                .acceleration_structure_device
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(buffer.buffer)
                        .offset(0)
                        .size(buffer.size)
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                    None,
                )?;

            let scratch_buffer: Buffer = self.create_buffer(
                size_info.build_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
                Some("BLAS Scratch Buffer"),
            )?;

            let build_info = build_info
                .dst_acceleration_structure(acceleration_structure)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.address,
                });

            let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(primitive_count)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);

            let command_pool = self.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(render_queue.queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?;

            let [command_buffer] = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?
                .try_into()
                .map_err(|_| anyhow!("Failed to allocate BLAS build command buffer"))?;

            self.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            self.acceleration_structure_device
                .cmd_build_acceleration_structures(
                    command_buffer,
                    &[build_info],
                    &[&[build_range]],
                );

            let memory_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR);

            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default()
                    .memory_barriers(std::slice::from_ref(&memory_barrier)),
            );

            self.device.end_command_buffer(command_buffer)?;

            self.device.queue_submit(
                render_queue.queue,
                &[
                    vk::SubmitInfo::default()
                        .command_buffers(std::slice::from_ref(&command_buffer)),
                ],
                vk::Fence::null(),
            )?;

            render_queue.wait_idle();
            self.device.destroy_command_pool(command_pool, None);
            self.destroy_buffer(scratch_buffer);

            let device_address = self
                .acceleration_structure_device
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(acceleration_structure),
                );

            Ok(Blas {
                acceleration_structure,
                buffer,
                device_address,
            })
        }
    }

    pub fn destroy_blas(&self, blas: Blas) {
        unsafe {
            self.acceleration_structure_device
                .destroy_acceleration_structure(blas.acceleration_structure, None);

            self.destroy_buffer(blas.buffer);
        }
    }
}
