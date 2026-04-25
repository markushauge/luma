use anyhow::{Result, anyhow};
use ash::vk;
use bevy::{
    ecs::system::{
        SystemParamItem,
        lifetimeless::{SRes, SResMut},
    },
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use super::{
    blas::Blas,
    buffer::Buffer,
    render_asset::{RenderAsset, RenderAssets, sync_render_assets},
    render_device::RenderDevice,
    render_queue::RenderQueue,
    schedule::{Render, RenderStartup, RenderSystems},
};

pub struct MeshPlugin;

impl Plugin for MeshPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderAssets<GpuMesh>>()
            .add_systems(RenderStartup, create_mesh_info_buffer)
            .add_systems(
                Render,
                sync_render_assets::<GpuMesh>.in_set(RenderSystems::Prepare),
            );
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

pub struct GpuMesh {
    pub render_device: RenderDevice,
    pub vertex_buffer: Buffer<Vertex>,
    pub index_buffer: Buffer<u32>,
    pub mesh_index: u32,
    pub blas: Blas,
}

impl RenderAsset for GpuMesh {
    type SourceAsset = Mesh;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SResMut<MeshInfoBuffer>,
    );

    fn prepare(
        source_asset: &Self::SourceAsset,
        (render_device, render_queue, mesh_info_buffer): &mut SystemParamItem<Self::Param>,
        previous_asset: Option<&Self>,
    ) -> Result<Self> {
        if let Some(previous_asset) = previous_asset {
            mesh_info_buffer.remove(previous_asset.mesh_index)?;
        }

        let vertex_buffer = mesh_to_vertex_buffer(render_device, source_asset)?;
        let index_buffer = mesh_to_index_buffer(render_device, source_asset)?;

        let mesh_index = mesh_info_buffer.insert(MeshInfo {
            vertex_buffer_address: vertex_buffer.address,
            index_buffer_address: index_buffer.address,
        })?;

        let blas = render_device.create_blas(render_queue, &vertex_buffer, &index_buffer)?;
        let render_device = render_device.clone();

        Ok(Self {
            render_device,
            vertex_buffer,
            index_buffer,
            mesh_index,
            blas,
        })
    }
}

impl Drop for GpuMesh {
    fn drop(&mut self) {
        self.render_device
            .destroy_blas(std::mem::take(&mut self.blas));

        self.render_device
            .destroy_buffer(std::mem::take(&mut self.index_buffer));

        self.render_device
            .destroy_buffer(std::mem::take(&mut self.vertex_buffer));
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, Pod, Zeroable)]
pub struct MeshInfo {
    pub vertex_buffer_address: vk::DeviceAddress,
    pub index_buffer_address: vk::DeviceAddress,
}

#[derive(Resource)]
pub struct MeshInfoBuffer {
    pub render_device: RenderDevice,
    pub buffer: Buffer<MeshInfo>,
    pub count: u32,
    pub free_indices: Vec<u32>,
}

impl MeshInfoBuffer {
    pub fn new(render_device: RenderDevice) -> Result<Self> {
        let buffer = Self::allocate_buffer(&render_device, 256)?;

        Ok(Self {
            render_device,
            buffer,
            count: 0,
            free_indices: Vec::new(),
        })
    }

    pub fn insert(&mut self, mesh_info: MeshInfo) -> Result<u32> {
        let index = if let Some(free_index) = self.free_indices.pop() {
            free_index
        } else {
            let index = self.count;
            self.count += 1;
            self.ensure_capacity(self.count)?;
            index
        };

        *self.index_mut(index)? = mesh_info;
        Ok(index)
    }

    pub fn remove(&mut self, index: u32) -> Result<Option<MeshInfo>> {
        if index >= self.count {
            return Ok(None);
        }

        let mesh_info = std::mem::take(self.index_mut(index)?);
        self.free_indices.push(index);
        Ok(Some(mesh_info))
    }

    fn index_mut(&mut self, index: u32) -> Result<&mut MeshInfo> {
        Ok(&mut self.buffer.slice_mut()?[index as usize])
    }

    fn ensure_capacity(&mut self, required_capacity: u32) -> Result<()> {
        let old_capacity = self.buffer.len as u32;

        if required_capacity <= old_capacity {
            return Ok(());
        }

        let new_capacity = required_capacity.next_power_of_two().max(old_capacity * 2);
        let mut new_buffer = Self::allocate_buffer(&self.render_device, new_capacity)?;

        new_buffer.slice_mut()?[..self.buffer.len as usize]
            .copy_from_slice(self.buffer.slice_mut()?);

        self.render_device
            .destroy_buffer(std::mem::replace(&mut self.buffer, new_buffer));

        Ok(())
    }

    fn allocate_buffer(render_device: &RenderDevice, capacity: u32) -> Result<Buffer<MeshInfo>> {
        render_device
            .create_buffer(
                capacity as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
                Some("Mesh Info Buffer"),
            )
            .map_err(Into::into)
    }
}

impl Drop for MeshInfoBuffer {
    fn drop(&mut self) {
        self.render_device
            .destroy_buffer(std::mem::take(&mut self.buffer));
    }
}

fn create_mesh_info_buffer(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
) -> Result<(), BevyError> {
    let mesh_info_buffer = MeshInfoBuffer::new(render_device.clone())?;
    commands.insert_resource(mesh_info_buffer);
    Ok(())
}

fn mesh_to_vertex_buffer(render_device: &RenderDevice, mesh: &Mesh) -> Result<Buffer<Vertex>> {
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(positions)) => positions,
        _ => return Err(anyhow!("Mesh is missing required position attribute")),
    };

    let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(normals)) => normals,
        _ => return Err(anyhow!("Mesh is missing required normal attribute")),
    };

    let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
        Some(VertexAttributeValues::Float32x2(uvs)) => uvs,
        _ => return Err(anyhow!("Mesh is missing required UV attribute")),
    };

    let attributes = positions.iter().zip(normals.iter()).zip(uvs.iter());

    let mut vertex_buffer = render_device.create_buffer(
        attributes.len() as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::CpuToGpu,
        Some("Vertex Buffer"),
    )?;

    vertex_buffer
        .slice_mut()?
        .iter_mut()
        .zip(attributes)
        .for_each(|(slot, ((position, normal), uv))| {
            *slot = Vertex {
                position: Vec3::from(*position),
                normal: Vec3::from(*normal),
                uv: Vec2::from(*uv),
            };
        });

    Ok(vertex_buffer)
}

fn mesh_to_index_buffer(render_device: &RenderDevice, mesh: &Mesh) -> Result<Buffer<u32>> {
    let indices = match mesh.indices() {
        Some(Indices::U32(indices)) => indices,
        _ => return Err(anyhow!("Mesh is missing required indices")),
    };

    let mut index_buffer = render_device.create_buffer(
        indices.len() as u64,
        vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::CpuToGpu,
        Some("Index Buffer"),
    )?;

    index_buffer.slice_mut()?.copy_from_slice(indices);
    Ok(index_buffer)
}
