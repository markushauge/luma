use std::{collections::HashMap, mem::size_of};

use anyhow::{Result, anyhow};
use ash::vk;
use bevy::{
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
};
use gpu_allocator::MemoryLocation;

use super::{
    Renderer,
    buffer::Buffer,
    render_device::RenderDevice,
    render_queue::RenderQueue,
    schedule::{Render, RenderStartup},
};

pub struct AccelerationStructurePlugin;

impl Plugin for AccelerationStructurePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(RenderStartup, create_acceleration_structure_manager)
            .add_systems(Render, (build_blases, build_tlas).chain());
    }
}

fn create_acceleration_structure_manager(mut commands: Commands, renderer: Res<Renderer>) {
    commands.insert_resource(AccelerationStructureManager::new(
        renderer.render_device.clone(),
    ));
}

fn build_blases(
    renderer: Res<Renderer>,
    mut manager: ResMut<AccelerationStructureManager>,
    meshes: Res<Assets<Mesh>>,
    mut asset_events: MessageReader<AssetEvent<Mesh>>,
) {
    for asset_event in asset_events.read() {
        if let AssetEvent::Added { id } = asset_event {
            let Some(mesh) = meshes.get(*id) else {
                tracing::error!("Mesh with ID {} not found in Assets<Mesh>.", id);
                continue;
            };

            if let Err(err) =
                manager.insert_mesh(&renderer.render_device, &renderer.render_queue, *id, mesh)
            {
                tracing::error!("{}", err);
                continue;
            }

            manager.tlas_dirty = true;
        }
    }
}

fn build_tlas(
    renderer: Res<Renderer>,
    mut manager: ResMut<AccelerationStructureManager>,
    query: Query<(&Transform, &Mesh3d)>,
    changed_mesh3d: Query<(), Changed<Mesh3d>>,
    changed_transform: Query<(), (Changed<Transform>, With<Mesh3d>)>,
    removed_mesh3d: RemovedComponents<Mesh3d>,
) {
    manager.tlas_dirty |=
        !changed_mesh3d.is_empty() || !changed_transform.is_empty() || !removed_mesh3d.is_empty();

    if !manager.tlas_dirty {
        return;
    }

    let instances: Vec<_> = query
        .iter()
        .filter_map(|(transform, Mesh3d(mesh_handle))| {
            let blas = manager.blases.get(&mesh_handle.id())?;
            Some(BlasInstance {
                blas,
                transform: *transform,
            })
        })
        .collect();

    if instances.is_empty() {
        return;
    }

    let tlas = match renderer
        .render_device
        .create_tlas(&renderer.render_queue, &instances)
    {
        Ok(tlas) => tlas,
        Err(err) => {
            tracing::error!("{}", err);
            return;
        }
    };

    manager.tlas = Some(tlas);
    manager.tlas_dirty = false;
}

#[derive(Resource)]
pub struct AccelerationStructureManager {
    render_device: RenderDevice,
    blases: HashMap<AssetId<Mesh>, Blas>,
    tlas: Option<Tlas>,
    tlas_dirty: bool,
}

impl AccelerationStructureManager {
    fn new(render_device: RenderDevice) -> Self {
        Self {
            render_device,
            blases: HashMap::new(),
            tlas: None,
            tlas_dirty: false,
        }
    }

    fn insert_mesh(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        asset_id: AssetId<Mesh>,
        mesh: &Mesh,
    ) -> Result<()> {
        let Some(VertexAttributeValues::Float32x3(vertices)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            anyhow::bail!("Mesh does not contain [f32; 3] positions.");
        };

        let Some(Indices::U32(indices)) = mesh.indices() else {
            anyhow::bail!("Mesh does not contain u32 indices.");
        };

        let blas = render_device.create_blas(render_queue, vertices, indices)?;
        self.blases.insert(asset_id, blas);
        Ok(())
    }

    pub fn tlas(&self) -> Option<&Tlas> {
        self.tlas.as_ref()
    }
}

impl Drop for AccelerationStructureManager {
    fn drop(&mut self) {
        unsafe {
            for (_, blas) in std::mem::take(&mut self.blases) {
                self.render_device.destroy_blas(blas);
            }

            if let Some(tlas) = self.tlas.take() {
                self.render_device.destroy_tlas(tlas);
            }
        }
    }
}

#[derive(Resource)]
pub struct Blas {
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
        vertices: &[[f32; 3]],
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

            let vertex_buffer_address = self.get_buffer_device_address(&vertex_buffer);
            let index_buffer_address = self.get_buffer_device_address(&index_buffer);

            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: vertex_buffer_address,
                        })
                        .vertex_stride(size_of::<[f32; 3]>() as u64)
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

            let scratch_buffer = self.create_buffer(
                size_info.build_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
                Some("BLAS Scratch Buffer"),
            )?;

            let scratch_address = self.get_buffer_device_address(&scratch_buffer);

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

            self.device.queue_wait_idle(render_queue.queue)?;
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
                vertex_buffer,
                index_buffer,
            })
        }
    }

    pub unsafe fn destroy_blas(&self, blas: Blas) {
        unsafe {
            self.acceleration_structure_device
                .destroy_acceleration_structure(blas.acceleration_structure, None);
            self.destroy_buffer(blas.buffer);
            self.destroy_buffer(blas.index_buffer);
            self.destroy_buffer(blas.vertex_buffer);
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

            let mut instance_buffer = self.create_buffer(
                instance_buffer_size as u64,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                MemoryLocation::CpuToGpu,
                Some("TLAS Instance Buffer"),
            )?;

            // Safety: AccelerationStructureInstanceKHR is repr(C)
            let instance_bytes =
                std::slice::from_raw_parts(instances.as_ptr().cast::<u8>(), instance_buffer_size);

            instance_buffer.slice_mut()?.copy_from_slice(instance_bytes);

            let instance_buffer_address = self.get_buffer_device_address(&instance_buffer);

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

            let scratch_buffer = self.create_buffer(
                size_info.build_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
                Some("TLAS Scratch Buffer"),
            )?;

            let scratch_address = self.get_buffer_device_address(&scratch_buffer);

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

            self.device.queue_wait_idle(render_queue.queue)?;
            self.device.destroy_command_pool(command_pool, None);
            self.destroy_buffer(scratch_buffer);

            Ok(Tlas {
                acceleration_structure,
                buffer,
                instance_buffer,
            })
        }
    }

    pub unsafe fn destroy_tlas(&self, tlas: Tlas) {
        unsafe {
            self.acceleration_structure_device
                .destroy_acceleration_structure(tlas.acceleration_structure, None);
            self.destroy_buffer(tlas.buffer);
            self.destroy_buffer(tlas.instance_buffer);
        }
    }
}
