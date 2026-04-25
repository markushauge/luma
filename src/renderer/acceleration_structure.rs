use std::{collections::HashMap, mem::size_of};

use anyhow::{Result, anyhow};
use ash::vk;
use bevy::{
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use super::{
    buffer::Buffer,
    render_device::RenderDevice,
    render_queue::RenderQueue,
    schedule::{Render, RenderSystems},
};

pub struct AccelerationStructurePlugin;

impl Plugin for AccelerationStructurePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BlasManager>().add_systems(
            Render,
            build_acceleration_structures.in_set(RenderSystems::Prepare),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn build_acceleration_structures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut blas_manager: ResMut<BlasManager>,
    tlas: Option<ResMut<Tlas>>,
    meshes: Res<Assets<Mesh>>,
    mut asset_events: MessageReader<AssetEvent<Mesh>>,
    mesh3ds: Query<(&Transform, &Mesh3d)>,
    changed_mesh3ds: Query<(), Changed<Mesh3d>>,
    changed_transforms: Query<(), (Changed<Transform>, With<Mesh3d>)>,
    removed_mesh3ds: RemovedComponents<Mesh3d>,
) -> Result<(), BevyError> {
    let mut build_tlas = tlas.is_none()
        | !changed_mesh3ds.is_empty()
        | !changed_transforms.is_empty()
        | !removed_mesh3ds.is_empty();

    for asset_event in asset_events.read() {
        match asset_event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => {
                let Some(mesh) = meshes.get(*id) else {
                    continue;
                };

                let Some(vertices) = mesh_to_vertices(mesh) else {
                    tracing::error!("Mesh is missing required vertex attributes.");
                    continue;
                };

                let Some(Indices::U32(indices)) = mesh.indices() else {
                    tracing::error!("Mesh does not contain u32 indices.");
                    continue;
                };

                let blas = render_device.create_blas(&render_queue, &vertices, indices)?;
                blas_manager.blases.insert(*id, blas);
                build_tlas = true;
            }
            AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                blas_manager.blases.remove(id);
                build_tlas = true;
            }
        }
    }

    if !build_tlas {
        return Ok(());
    }

    let instances: Vec<_> = mesh3ds
        .iter()
        .filter_map(|(transform, Mesh3d(mesh_handle))| {
            let blas = blas_manager.blases.get(&mesh_handle.id())?;
            Some(BlasInstance {
                blas,
                transform: *transform,
            })
        })
        .collect();

    let tlas = render_device.create_tlas(&render_queue, &instances)?;
    commands.insert_resource(tlas);
    Ok(())
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

fn mesh_to_vertices(mesh: &Mesh) -> Option<Vec<Vertex>> {
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION)? {
        VertexAttributeValues::Float32x3(positions) => positions,
        _ => return None,
    };

    let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL)? {
        VertexAttributeValues::Float32x3(normals) => normals,
        _ => return None,
    };

    let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0)? {
        VertexAttributeValues::Float32x2(uvs) => uvs,
        _ => return None,
    };

    let vertices = positions
        .iter()
        .zip(normals.iter())
        .zip(uvs.iter())
        .map(|((position, normal), uv)| Vertex {
            position: Vec3::from(*position),
            normal: Vec3::from(*normal),
            uv: Vec2::from(*uv),
        })
        .collect();

    Some(vertices)
}

#[derive(Resource, Default)]
pub struct BlasManager {
    blases: HashMap<AssetId<Mesh>, Blas>,
}

#[derive(Resource)]
pub struct Blas {
    render_device: RenderDevice,
    acceleration_structure: vk::AccelerationStructureKHR,
    buffer: Buffer,
    device_address: vk::DeviceAddress,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
}

pub struct BlasInstance<'a> {
    pub blas: &'a Blas,
    pub transform: Transform,
}

#[derive(Resource)]
pub struct Tlas {
    render_device: RenderDevice,
    acceleration_structure: vk::AccelerationStructureKHR,
    buffer: Buffer,
    instance_buffer: Buffer,
}

impl Tlas {
    pub fn acceleration_structure(&self) -> &vk::AccelerationStructureKHR {
        &self.acceleration_structure
    }
}

impl RenderDevice {
    pub fn create_blas(
        &self,
        render_queue: &RenderQueue,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Result<Blas> {
        unsafe {
            let mut vertex_buffer = self.create_buffer(
                size_of_val(vertices) as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                MemoryLocation::CpuToGpu,
                Some("BLAS Vertex Buffer"),
            )?;

            vertex_buffer
                .slice_mut()?
                .copy_from_slice(bytemuck::cast_slice(vertices));

            let mut index_buffer = self.create_buffer(
                size_of_val(indices) as u64,
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                MemoryLocation::CpuToGpu,
                Some("BLAS Index Buffer"),
            )?;

            index_buffer
                .slice_mut()?
                .copy_from_slice(bytemuck::cast_slice(indices));

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
                        .max_vertex(vertices.len() as u32 - 1)
                        .index_type(vk::IndexType::UINT32)
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: index_buffer.address,
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
                render_device: self.clone(),
                acceleration_structure,
                buffer,
                device_address,
                vertex_buffer,
                index_buffer,
            })
        }
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        unsafe {
            self.render_device
                .acceleration_structure_device
                .destroy_acceleration_structure(self.acceleration_structure, None);

            self.render_device
                .destroy_buffer(std::mem::take(&mut self.buffer));

            self.render_device
                .destroy_buffer(std::mem::take(&mut self.vertex_buffer));

            self.render_device
                .destroy_buffer(std::mem::take(&mut self.index_buffer));
        }
    }
}

impl RenderDevice {
    pub fn create_tlas(
        &self,
        render_queue: &RenderQueue,
        instances: &[BlasInstance],
    ) -> Result<Tlas> {
        unsafe {
            let instances: Vec<vk::AccelerationStructureInstanceKHR> = instances
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

            let instance_count = instances.len() as u32;
            let instance_buffer_size =
                instances.len() * size_of::<vk::AccelerationStructureInstanceKHR>();

            let instance_buffer = if instance_buffer_size > 0 {
                let mut instance_buffer = self.create_buffer(
                    instance_buffer_size as u64,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    MemoryLocation::CpuToGpu,
                    Some("TLAS Instance Buffer"),
                )?;

                // Safety: AccelerationStructureInstanceKHR is repr(C)
                let instance_bytes = std::slice::from_raw_parts(
                    instances.as_ptr().cast::<u8>(),
                    instance_buffer_size,
                );

                instance_buffer.slice_mut()?.copy_from_slice(instance_bytes);
                instance_buffer
            } else {
                Buffer::default()
            };

            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                        .array_of_pointers(false)
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: instance_buffer.address,
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

            let buffer = self.create_buffer(
                size_info.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
                Some("TLAS Buffer"),
            )?;

            let acceleration_structure = self
                .acceleration_structure_device
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(buffer.buffer)
                        .offset(0)
                        .size(buffer.size)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                    None,
                )?;

            let scratch_buffer: Buffer = self.create_buffer(
                size_info.build_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
                Some("TLAS Scratch Buffer"),
            )?;

            let build_info = build_info
                .dst_acceleration_structure(acceleration_structure)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.address,
                });

            let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(instance_count)
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

            Ok(Tlas {
                render_device: self.clone(),
                acceleration_structure,
                buffer,
                instance_buffer,
            })
        }
    }
}

impl Drop for Tlas {
    fn drop(&mut self) {
        unsafe {
            self.render_device
                .acceleration_structure_device
                .destroy_acceleration_structure(self.acceleration_structure, None);

            self.render_device
                .destroy_buffer(std::mem::take(&mut self.buffer));

            self.render_device
                .destroy_buffer(std::mem::take(&mut self.instance_buffer));
        }
    }
}
