use anyhow::{Result, anyhow};
use ash::vk;
use bevy::{math::Affine3A, prelude::*};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use super::{
    blas::Blas,
    buffer::Buffer,
    mesh::GpuMesh,
    render_asset::{RenderAssets, sync_render_assets},
    render_device::RenderDevice,
    render_queue::RenderQueue,
    schedule::{Render, RenderSystems},
};

pub struct TlasPlugin;

impl Plugin for TlasPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Render,
            build_tlas
                .in_set(RenderSystems::Prepare)
                .after(sync_render_assets::<GpuMesh>),
        );
    }
}

#[expect(clippy::too_many_arguments)]
fn build_tlas(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_meshes: Res<RenderAssets<GpuMesh>>,
    tlas: Option<ResMut<Tlas>>,
    mut asset_events: MessageReader<AssetEvent<Mesh>>,
    mesh3ds: Query<(&GlobalTransform, &Mesh3d)>,
    changed_mesh3ds: Query<(), Changed<Mesh3d>>,
    changed_transforms: Query<(), (Changed<GlobalTransform>, With<Mesh3d>)>,
    removed_mesh3ds: RemovedComponents<Mesh3d>,
) -> Result<(), BevyError> {
    let build_tlas = tlas.is_none()
        | !changed_mesh3ds.is_empty()
        | !changed_transforms.is_empty()
        | !removed_mesh3ds.is_empty()
        | !asset_events.is_empty();

    asset_events.clear();

    if !build_tlas {
        return Ok(());
    }

    let instances: Vec<_> = mesh3ds
        .iter()
        .filter_map(|(transform, Mesh3d(mesh_handle))| {
            let mesh = gpu_meshes.get(&mesh_handle.id())?;

            Some(BlasInstance {
                mesh_index: mesh.mesh_index,
                blas: &mesh.blas,
                transform: transform.affine(),
            })
        })
        .collect();

    let tlas = render_device.create_tlas(&render_queue, &instances)?;
    commands.insert_resource(tlas);
    Ok(())
}

pub struct BlasInstance<'a> {
    pub mesh_index: u32,
    pub blas: &'a Blas,
    pub transform: Affine3A,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AccelerationStructureInstance(vk::AccelerationStructureInstanceKHR);

// Safety: AccelerationStructureInstanceKHR is repr(C)
unsafe impl Pod for AccelerationStructureInstance {}
unsafe impl Zeroable for AccelerationStructureInstance {}

impl From<&BlasInstance<'_>> for AccelerationStructureInstance {
    fn from(instance: &BlasInstance) -> Self {
        let x_axis = instance.transform.x_axis;
        let y_axis = instance.transform.y_axis;
        let z_axis = instance.transform.z_axis;
        let w_axis = instance.transform.w_axis;

        let transform = vk::TransformMatrixKHR {
            matrix: [
                x_axis.x, y_axis.x, z_axis.x, w_axis.x, // X
                x_axis.y, y_axis.y, z_axis.y, w_axis.y, // Y
                x_axis.z, y_axis.z, z_axis.z, w_axis.z, // Z
            ],
        };

        Self(vk::AccelerationStructureInstanceKHR {
            transform,
            instance_custom_index_and_mask: vk::Packed24_8::new(instance.mesh_index, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as _,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: instance.blas.device_address,
            },
        })
    }
}

#[derive(Resource)]
pub struct Tlas {
    render_device: RenderDevice,
    buffer: Buffer,
    acceleration_structure: vk::AccelerationStructureKHR,
}

impl Tlas {
    pub fn acceleration_structure(&self) -> &vk::AccelerationStructureKHR {
        &self.acceleration_structure
    }
}

impl RenderDevice {
    pub fn create_tlas(
        &self,
        render_queue: &RenderQueue,
        instances: &[BlasInstance],
    ) -> Result<Tlas> {
        unsafe {
            let instance_count = instances.len() as u32;

            let instance_buffer = if instance_count > 0 {
                let mut instance_buffer = self.create_buffer::<AccelerationStructureInstance>(
                    instance_count as u64,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    MemoryLocation::CpuToGpu,
                    Some("TLAS Instance Buffer"),
                )?;

                instance_buffer
                    .slice_mut()?
                    .iter_mut()
                    .zip(instances)
                    .for_each(|(slot, instance)| {
                        *slot = AccelerationStructureInstance::from(instance);
                    });

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
            self.destroy_buffer(instance_buffer);

            Ok(Tlas {
                render_device: self.clone(),
                acceleration_structure,
                buffer,
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
        }
    }
}
