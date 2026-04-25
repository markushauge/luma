use anyhow::{Result, anyhow};
use ash::vk;
use bevy::{
    ecs::system::{SystemParamItem, lifetimeless::SRes},
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
    schedule::{Render, RenderSystems},
};

pub struct MeshPlugin;

impl Plugin for MeshPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderAssets<GpuMesh>>().add_systems(
            Render,
            sync_render_assets::<GpuMesh>.in_set(RenderSystems::PrepareAssets),
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
    pub blas: Blas,
}

impl RenderAsset for GpuMesh {
    type SourceAsset = Mesh;
    type Param = (SRes<RenderDevice>, SRes<RenderQueue>);

    fn prepare(
        mesh: &Self::SourceAsset,
        (render_device, render_queue): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self> {
        let vertex_buffer = mesh_to_vertex_buffer(render_device, mesh)?;
        let index_buffer = mesh_to_index_buffer(render_device, mesh)?;
        let blas = render_device.create_blas(render_queue, &vertex_buffer, &index_buffer)?;
        let render_device = render_device.clone();

        Ok(Self {
            render_device,
            vertex_buffer,
            index_buffer,
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
        .for_each(|(vertex, ((position, normal), uv))| {
            *vertex = Vertex {
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
