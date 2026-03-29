use std::mem::size_of;

use anyhow::{Result, anyhow};
use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use super::Device;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
}

#[derive(Resource)]
pub struct Blas {
    pub acceleration_structure: vk::AccelerationStructureKHR,
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub device_address: vk::DeviceAddress,
    vertex_buffer: vk::Buffer,
    vertex_allocation: Allocation,
    index_buffer: vk::Buffer,
    index_allocation: Allocation,
}

pub struct BlasInstance<'a> {
    pub blas: &'a Blas,
    pub transform: Transform,
}

#[derive(Resource)]
pub struct Tlas {
    pub acceleration_structure: vk::AccelerationStructureKHR,
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    instance_buffer: vk::Buffer,
    instance_allocation: Allocation,
}

impl Device {
    pub fn create_blas(&self, vertices: &[Vertex], indices: &[u32]) -> Result<Blas> {
        unsafe {
            let vertex_buffer_size = size_of_val(vertices) as u64;
            let vertex_buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(vertex_buffer_size)
                    .usage(
                        vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let vertex_requirements = self.device.get_buffer_memory_requirements(vertex_buffer);

            let mut vertex_allocation = self.allocate(&AllocationCreateDesc {
                name: "BLAS Vertex Buffer",
                requirements: vertex_requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(vertex_buffer),
            })?;

            self.device.bind_buffer_memory(
                vertex_buffer,
                vertex_allocation.memory(),
                vertex_allocation.offset(),
            )?;

            let vertex_slice = vertex_allocation
                .mapped_slice_mut()
                .ok_or_else(|| anyhow!("Buffer is not host visible"))?;
            vertex_slice[..vertex_buffer_size as usize]
                .copy_from_slice(bytemuck::cast_slice(vertices));

            let index_buffer_size = size_of_val(indices) as u64;
            let index_buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(index_buffer_size)
                    .usage(
                        vk::BufferUsageFlags::INDEX_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let index_requirements = self.device.get_buffer_memory_requirements(index_buffer);

            let mut index_allocation = self.allocate(&AllocationCreateDesc {
                name: "BLAS Index Buffer",
                requirements: index_requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(index_buffer),
            })?;

            self.device.bind_buffer_memory(
                index_buffer,
                index_allocation.memory(),
                index_allocation.offset(),
            )?;

            let index_slice = index_allocation
                .mapped_slice_mut()
                .ok_or_else(|| anyhow!("Buffer is not host visible"))?;
            index_slice[..index_buffer_size as usize]
                .copy_from_slice(bytemuck::cast_slice(indices));

            let vertex_buffer_address = self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer),
            );

            let index_buffer_address = self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(index_buffer),
            );

            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: vertex_buffer_address,
                        })
                        .vertex_stride(size_of::<Vertex>() as u64)
                        .max_vertex(vertices.len() as u32 - 1)
                        .index_type(vk::IndexType::UINT32)
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: index_buffer_address,
                        }),
                });

            let primitive_count = (indices.len() / 3) as u32;

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

            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size_info.acceleration_structure_size)
                    .usage(
                        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let allocation = self.allocate(&AllocationCreateDesc {
                name: "BLAS",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
            })?;

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;

            let acceleration_structure = self
                .acceleration_structure_device
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(buffer)
                        .offset(0)
                        .size(size_info.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                    None,
                )?;

            let scratch_buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size_info.build_scratch_size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let scratch_requirements = self.device.get_buffer_memory_requirements(scratch_buffer);
            let scratch_allocation = self.allocate(&AllocationCreateDesc {
                name: "BLAS Scratch Buffer",
                requirements: scratch_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(scratch_buffer),
            })?;

            self.device.bind_buffer_memory(
                scratch_buffer,
                scratch_allocation.memory(),
                scratch_allocation.offset(),
            )?;

            let scratch_address = self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer),
            );

            let build_info = build_info
                .dst_acceleration_structure(acceleration_structure)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_address,
                });

            let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(primitive_count)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);

            let command_pool = self.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.queue_family_index)
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

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)],
                &[],
                &[],
            );

            self.device.end_command_buffer(command_buffer)?;

            self.device.queue_submit(
                self.queue,
                &[
                    vk::SubmitInfo::default()
                        .command_buffers(std::slice::from_ref(&command_buffer)),
                ],
                vk::Fence::null(),
            )?;

            self.device.queue_wait_idle(self.queue)?;
            self.device.destroy_command_pool(command_pool, None);
            self.device.destroy_buffer(scratch_buffer, None);
            self.free(scratch_allocation)?;

            let device_address = self
                .acceleration_structure_device
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(acceleration_structure),
                );

            Ok(Blas {
                acceleration_structure,
                buffer,
                allocation,
                device_address,
                vertex_buffer,
                vertex_allocation,
                index_buffer,
                index_allocation,
            })
        }
    }
}

impl Blas {
    #[expect(dead_code)]
    pub unsafe fn destroy(self, device: &Device) {
        unsafe {
            device
                .acceleration_structure_device
                .destroy_acceleration_structure(self.acceleration_structure, None);
            device.device.destroy_buffer(self.buffer, None);
            device.free(self.allocation).unwrap();
            device.device.destroy_buffer(self.index_buffer, None);
            device.free(self.index_allocation).unwrap();
            device.device.destroy_buffer(self.vertex_buffer, None);
            device.free(self.vertex_allocation).unwrap();
        }
    }
}

impl Device {
    pub fn create_tlas(&self, instances: &[BlasInstance<'_>]) -> Result<Tlas> {
        unsafe {
            let vk_instances: Vec<vk::AccelerationStructureInstanceKHR> = instances
                .iter()
                .map(|instance| {
                    let affine = instance.transform.compute_affine();
                    let x_axis = affine.x_axis;
                    let y_axis = affine.y_axis;
                    let z_axis = affine.z_axis;
                    let w_axis = affine.w_axis;

                    let transform = vk::TransformMatrixKHR {
                        matrix: [
                            x_axis.x, y_axis.x, z_axis.x, w_axis.x, // X
                            x_axis.y, y_axis.y, z_axis.y, w_axis.y, // Y
                            x_axis.z, y_axis.z, z_axis.z, w_axis.z, // Z
                        ],
                    };

                    vk::AccelerationStructureInstanceKHR {
                        transform,
                        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
                        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                            0,
                            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw()
                                as _,
                        ),
                        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                            device_handle: instance.blas.device_address,
                        },
                    }
                })
                .collect();

            let instance_count = vk_instances.len() as u32;
            let instance_buffer_size =
                (vk_instances.len() * size_of::<vk::AccelerationStructureInstanceKHR>()) as u64;

            let instance_buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(instance_buffer_size)
                    .usage(
                        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let instance_requirements = self.device.get_buffer_memory_requirements(instance_buffer);

            let mut instance_allocation = self.allocate(&AllocationCreateDesc {
                name: "TLAS Instance Buffer",
                requirements: instance_requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(instance_buffer),
            })?;

            self.device.bind_buffer_memory(
                instance_buffer,
                instance_allocation.memory(),
                instance_allocation.offset(),
            )?;

            let instance_slice = instance_allocation
                .mapped_slice_mut()
                .ok_or_else(|| anyhow!("Instance buffer is not mapped"))?;

            // Safety: AccelerationStructureInstanceKHR is repr(C)
            let instance_bytes = std::slice::from_raw_parts(
                vk_instances.as_ptr().cast::<u8>(),
                instance_buffer_size as usize,
            );
            instance_slice[..instance_buffer_size as usize].copy_from_slice(instance_bytes);

            let instance_buffer_address = self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(instance_buffer),
            );

            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                        .array_of_pointers(false)
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: instance_buffer_address,
                        }),
                });

            let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(&geometry));

            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            self.acceleration_structure_device
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[instance_count],
                    &mut size_info,
                );

            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size_info.acceleration_structure_size)
                    .usage(
                        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let allocation = self.allocate(&AllocationCreateDesc {
                name: "TLAS",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
            })?;

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;

            let acceleration_structure = self
                .acceleration_structure_device
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(buffer)
                        .offset(0)
                        .size(size_info.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                    None,
                )?;

            let scratch_buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size_info.build_scratch_size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let scratch_requirements = self.device.get_buffer_memory_requirements(scratch_buffer);
            let scratch_allocation = self.allocate(&AllocationCreateDesc {
                name: "TLAS Scratch Buffer",
                requirements: scratch_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(scratch_buffer),
            })?;

            self.device.bind_buffer_memory(
                scratch_buffer,
                scratch_allocation.memory(),
                scratch_allocation.offset(),
            )?;

            let scratch_address = self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer),
            );

            let build_info = build_info
                .dst_acceleration_structure(acceleration_structure)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_address,
                });

            let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(instance_count)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);

            let command_pool = self.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.queue_family_index)
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
                .map_err(|_| anyhow!("Failed to allocate TLAS build command buffer"))?;

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

            self.device.end_command_buffer(command_buffer)?;

            self.device.queue_submit(
                self.queue,
                &[
                    vk::SubmitInfo::default()
                        .command_buffers(std::slice::from_ref(&command_buffer)),
                ],
                vk::Fence::null(),
            )?;

            self.device.queue_wait_idle(self.queue)?;
            self.device.destroy_command_pool(command_pool, None);
            self.device.destroy_buffer(scratch_buffer, None);
            self.free(scratch_allocation)?;

            Ok(Tlas {
                acceleration_structure,
                buffer,
                allocation,
                instance_buffer,
                instance_allocation,
            })
        }
    }
}

impl Tlas {
    #[expect(dead_code)]
    pub unsafe fn destroy(self, device: &Device) {
        unsafe {
            device
                .acceleration_structure_device
                .destroy_acceleration_structure(self.acceleration_structure, None);
            device.device.destroy_buffer(self.buffer, None);
            device.free(self.allocation).unwrap();
            device.device.destroy_buffer(self.instance_buffer, None);
            device.free(self.instance_allocation).unwrap();
        }
    }
}
