#![allow(dead_code)]

use std::mem::size_of;

use anyhow::{Result, anyhow};
use ash::vk::{self, Extent2D};
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use crate::{
    camera::Camera,
    renderer::{
        Renderer,
        schedule::{Render, RenderSystems},
    },
    shader::Shader,
};

use super::Device;

#[derive(Default)]
pub struct RayTracingPlugin {
    pub settings: RayTracingSettings,
}

impl Plugin for RayTracingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.settings.clone())
            .add_systems(Startup, load_shaders)
            .add_systems(
                Render,
                (
                    create_or_update_ray_tracing_pipeline,
                    execute_ray_tracing_pipeline,
                )
                    .chain()
                    .in_set(RenderSystems::Render),
            );
    }
}

#[derive(Resource, Clone)]
pub struct RayTracingSettings {
    pub resolution_scaling: f32,
}

impl Default for RayTracingSettings {
    fn default() -> Self {
        Self {
            resolution_scaling: 1.0,
        }
    }
}

#[derive(Resource)]
struct RayTracingShaders {
    raygen: Handle<Shader>,
    miss: Handle<Shader>,
    closest_hit: Handle<Shader>,
}

fn load_shaders(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(RayTracingShaders {
        raygen: asset_server.load("shaders/raygen.rgen"),
        miss: asset_server.load("shaders/miss.rmiss"),
        closest_hit: asset_server.load("shaders/closest_hit.rchit"),
    });
}

fn create_or_update_ray_tracing_pipeline(
    mut commands: Commands,
    renderer: Res<Renderer>,
    ray_tracing_pipeline: Option<ResMut<RayTracingPipeline>>,
    settings: Res<RayTracingSettings>,
    ray_tracing_shaders: Res<RayTracingShaders>,
    assets: Res<Assets<Shader>>,
) -> Result<(), BevyError> {
    let Some(raygen_shader) = assets.get(&ray_tracing_shaders.raygen) else {
        return Ok(());
    };

    let Some(miss_shader) = assets.get(&ray_tracing_shaders.miss) else {
        return Ok(());
    };

    let Some(hit_shader) = assets.get(&ray_tracing_shaders.closest_hit) else {
        return Ok(());
    };

    let Extent2D { width, height } = renderer.swapchain.surface_extent;
    let expected_width = (width as f32 * settings.resolution_scaling) as u32;
    let expected_height = (height as f32 * settings.resolution_scaling) as u32;

    match ray_tracing_pipeline {
        None => {
            commands.insert_resource(RayTracingPipeline::new(
                renderer.device.clone(),
                raygen_shader,
                miss_shader,
                hit_shader,
                expected_width,
                expected_height,
            )?);
        }
        Some(mut ray_tracing_pipeline) => {
            let vk::Extent3D { width, height, .. } = ray_tracing_pipeline.storage_image_extent;

            if width != expected_width || height != expected_height {
                ray_tracing_pipeline.recreate_storage_image(expected_width, expected_height)?;
            }
        }
    }

    Ok(())
}

fn execute_ray_tracing_pipeline(
    renderer: Res<Renderer>,
    ray_tracing_pipeline: Option<Res<RayTracingPipeline>>,
    camera: Query<(&Camera, &Transform), With<Camera>>,
    time: Res<Time>,
) -> Result<(), BevyError> {
    let Some(ray_tracing_pipeline) = ray_tracing_pipeline else {
        return Ok(());
    };

    let (camera, camera_transform) = camera.single()?;

    ray_tracing_pipeline.trace_rays(
        renderer.command_buffer,
        camera_transform,
        camera.vertical_fov(),
        time.elapsed().as_millis() as u32,
    );

    ray_tracing_pipeline.blit(
        renderer.command_buffer,
        renderer.swapchain.present_image().image,
        renderer.swapchain.surface_extent,
    );

    Ok(())
}

#[derive(Resource)]
pub struct RayTracingPipeline {
    pub device: Device,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub shader_binding_table: vk::Buffer,
    pub shader_binding_table_allocation: Allocation,
    pub sbt_raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub sbt_miss_region: vk::StridedDeviceAddressRegionKHR,
    pub sbt_hit_region: vk::StridedDeviceAddressRegionKHR,
    pub sbt_callable_region: vk::StridedDeviceAddressRegionKHR,
    pub storage_image: vk::Image,
    pub storage_image_view: vk::ImageView,
    pub storage_image_allocation: Allocation,
    pub storage_image_extent: vk::Extent3D,
    pub vertex_buffer: vk::Buffer,
    pub vertex_allocation: Allocation,
    pub index_buffer: vk::Buffer,
    pub index_allocation: Allocation,
    pub blas: vk::AccelerationStructureKHR,
    pub blas_buffer: vk::Buffer,
    pub blas_allocation: Allocation,
    pub tlas: vk::AccelerationStructureKHR,
    pub tlas_buffer: vk::Buffer,
    pub tlas_allocation: Allocation,
    pub instance_buffer: vk::Buffer,
    pub instance_allocation: Allocation,
    pub descriptor_set: vk::DescriptorSet,
}

impl RayTracingPipeline {
    pub fn new(
        device: Device,
        raygen_shader: &Shader,
        miss_shader: &Shader,
        hit_shader: &Shader,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        unsafe {
            let raygen_shader_module = device.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&raygen_shader.code),
                None,
            )?;

            let miss_shader_module = device.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&miss_shader.code),
                None,
            )?;

            let hit_shader_module = device.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&hit_shader.code),
                None,
            )?;

            // Define shader stages
            let shader_stages = vec![
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                    .module(raygen_shader_module)
                    .name(&raygen_shader.entry_point),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::MISS_KHR)
                    .module(miss_shader_module)
                    .name(&miss_shader.entry_point),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                    .module(hit_shader_module)
                    .name(&hit_shader.entry_point),
            ];

            let shader_modules = vec![raygen_shader_module, miss_shader_module, hit_shader_module];

            let shader_groups = vec![
                // Group 0: Raygen
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(0)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
                // Group 1: Miss
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(1)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
                // Group 2: Hit
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(2)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
            ];

            let descriptor_set_layout_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            ];

            let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&descriptor_set_layout_bindings);

            let descriptor_set_layout = device
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?;

            let push_constant_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                .offset(0)
                .size(size_of::<PushConstants>() as u32);

            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(std::slice::from_ref(&push_constant_range));

            let pipeline_layout = device
                .device
                .create_pipeline_layout(&pipeline_layout_create_info, None)?;

            let pipeline_create_info = vk::RayTracingPipelineCreateInfoKHR::default()
                .stages(&shader_stages)
                .groups(&shader_groups)
                .max_pipeline_ray_recursion_depth(1)
                .layout(pipeline_layout);

            let ray_tracing_pipeline_device = &device.ray_tracing_pipeline_device;

            let pipelines = ray_tracing_pipeline_device
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[pipeline_create_info],
                    None,
                )
                .map_err(|(_, result)| {
                    anyhow!("Failed to create ray tracing pipeline: {:?}", result)
                })?;

            let pipeline = pipelines
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("Failed to create ray tracing pipeline"))?;

            for module in shader_modules {
                device.device.destroy_shader_module(module, None);
            }

            let rt_props = device.get_physical_device_ray_tracing_pipeline_properties();
            let handle_size = rt_props.shader_group_handle_size as usize;
            let handle_alignment = rt_props.shader_group_handle_alignment as usize;
            let base_alignment = rt_props.shader_group_base_alignment as usize;

            let record_stride = handle_size.next_multiple_of(handle_alignment);

            // Calculate SBT regions (one record each: raygen, miss, hit)
            let raygen_region_offset = 0usize;
            let raygen_region_size = record_stride;

            let miss_region_offset =
                (raygen_region_offset + raygen_region_size).next_multiple_of(base_alignment);
            let miss_region_size = record_stride;

            let hit_region_offset =
                (miss_region_offset + miss_region_size).next_multiple_of(base_alignment);
            let hit_region_size = record_stride;

            let sbt_size = hit_region_offset + hit_region_size;

            let sbt_buffer_create_info = vk::BufferCreateInfo::default()
                .size(sbt_size as u64)
                .usage(
                    vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let shader_binding_table =
                device.device.create_buffer(&sbt_buffer_create_info, None)?;

            let requirements = device
                .device
                .get_buffer_memory_requirements(shader_binding_table);

            let mut shader_binding_table_allocation = device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing Pipeline Shader Binding Table",
                requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(shader_binding_table),
            })?;

            device.device.bind_buffer_memory(
                shader_binding_table,
                shader_binding_table_allocation.memory(),
                shader_binding_table_allocation.offset(),
            )?;

            let total_handle_size = handle_size * shader_groups.len();
            let shader_handles = ray_tracing_pipeline_device
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    shader_groups.len() as u32,
                    total_handle_size,
                )
                .map_err(|_| anyhow!("Failed to get shader group handles"))?;

            let sbt_data = shader_binding_table_allocation
                .mapped_slice_mut()
                .ok_or_else(|| anyhow!("Shader binding table is not mapped to host memory"))?;

            sbt_data[..sbt_size].fill(0);

            // Group 0: Raygen
            sbt_data[raygen_region_offset..raygen_region_offset + handle_size]
                .copy_from_slice(&shader_handles[0..handle_size]);

            // Group 1: Miss
            sbt_data[miss_region_offset..miss_region_offset + handle_size]
                .copy_from_slice(&shader_handles[handle_size..handle_size * 2]);

            // Group 2: Hit
            sbt_data[hit_region_offset..hit_region_offset + handle_size]
                .copy_from_slice(&shader_handles[handle_size * 2..handle_size * 3]);

            let sbt_address = device.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(shader_binding_table),
            );

            let sbt_raygen_region = vk::StridedDeviceAddressRegionKHR::default()
                .device_address(sbt_address + raygen_region_offset as u64)
                .stride(record_stride as u64)
                .size(raygen_region_size as u64);

            let sbt_miss_region = vk::StridedDeviceAddressRegionKHR::default()
                .device_address(sbt_address + miss_region_offset as u64)
                .stride(record_stride as u64)
                .size(miss_region_size as u64);

            let sbt_hit_region = vk::StridedDeviceAddressRegionKHR::default()
                .device_address(sbt_address + hit_region_offset as u64)
                .stride(record_stride as u64)
                .size(hit_region_size as u64);

            let sbt_callable_region = vk::StridedDeviceAddressRegionKHR::default();

            let storage_image_extent = vk::Extent3D {
                width,
                height,
                depth: 1,
            };

            let storage_image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(storage_image_extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let storage_image = device
                .device
                .create_image(&storage_image_create_info, None)?;

            let requirements = device.device.get_image_memory_requirements(storage_image);

            let storage_image_allocation = device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing Pipeline Storage Image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedImage(storage_image),
            })?;

            device.device.bind_image_memory(
                storage_image,
                storage_image_allocation.memory(),
                storage_image_allocation.offset(),
            )?;

            let storage_image_view_info = vk::ImageViewCreateInfo::default()
                .image(storage_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let storage_image_view = device
                .device
                .create_image_view(&storage_image_view_info, None)?;

            // Create cube mesh
            // Cube vertices: position (x, y, z)
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct Vertex {
                pos: [f32; 3],
            }

            let vertices = [
                // Front face
                Vertex {
                    pos: [-0.5, -0.5, -0.5],
                },
                Vertex {
                    pos: [0.5, -0.5, -0.5],
                },
                Vertex {
                    pos: [0.5, 0.5, -0.5],
                },
                Vertex {
                    pos: [-0.5, 0.5, -0.5],
                },
                // Back face
                Vertex {
                    pos: [-0.5, -0.5, 0.5],
                },
                Vertex {
                    pos: [0.5, -0.5, 0.5],
                },
                Vertex {
                    pos: [0.5, 0.5, 0.5],
                },
                Vertex {
                    pos: [-0.5, 0.5, 0.5],
                },
            ];

            let indices: [u32; 36] = [
                // Front
                0, 1, 2, 2, 3, 0, // Right
                1, 5, 6, 6, 2, 1, // Back
                5, 4, 7, 7, 6, 5, // Left
                4, 0, 3, 3, 7, 4, // Top
                3, 2, 6, 6, 7, 3, // Bottom
                4, 5, 1, 1, 0, 4,
            ];

            // Create vertex buffer
            let vertex_buffer_size = (vertices.len() * size_of::<Vertex>()) as u64;
            let vertex_buffer_create_info = vk::BufferCreateInfo::default()
                .size(vertex_buffer_size)
                .usage(
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let vertex_buffer = device
                .device
                .create_buffer(&vertex_buffer_create_info, None)?;
            let vertex_requirements = device.device.get_buffer_memory_requirements(vertex_buffer);

            let mut vertex_allocation = device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing Vertex Buffer",
                requirements: vertex_requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(vertex_buffer),
            })?;

            device.device.bind_buffer_memory(
                vertex_buffer,
                vertex_allocation.memory(),
                vertex_allocation.offset(),
            )?;

            // Copy vertex data
            let vertex_slice = vertex_allocation
                .mapped_slice_mut()
                .ok_or_else(|| anyhow!("Vertex buffer is not mapped"))?;
            vertex_slice[..vertex_buffer_size as usize]
                .copy_from_slice(bytemuck::cast_slice(&vertices));

            // Create index buffer
            let index_buffer_size = (indices.len() * size_of::<u32>()) as u64;
            let index_buffer_create_info = vk::BufferCreateInfo::default()
                .size(index_buffer_size)
                .usage(
                    vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let index_buffer = device
                .device
                .create_buffer(&index_buffer_create_info, None)?;
            let index_requirements = device.device.get_buffer_memory_requirements(index_buffer);

            let mut index_allocation = device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing Index Buffer",
                requirements: index_requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(index_buffer),
            })?;

            device.device.bind_buffer_memory(
                index_buffer,
                index_allocation.memory(),
                index_allocation.offset(),
            )?;

            // Copy index data
            let index_slice = index_allocation
                .mapped_slice_mut()
                .ok_or_else(|| anyhow!("Index buffer is not mapped"))?;
            index_slice[..index_buffer_size as usize]
                .copy_from_slice(bytemuck::cast_slice(&indices));

            // Get buffer device addresses
            let vertex_buffer_address = device.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer),
            );

            let index_buffer_address = device.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(index_buffer),
            );

            // Build BLAS
            let blas_geometry = vk::AccelerationStructureGeometryKHR::default()
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

            let blas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(&blas_geometry));

            let primitive_count = (indices.len() / 3) as u32;
            let mut blas_size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            device
                .acceleration_structure_device
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &blas_build_info,
                    &[primitive_count],
                    &mut blas_size_info,
                );

            // Create BLAS buffer
            let blas_buffer_create_info = vk::BufferCreateInfo::default()
                .size(blas_size_info.acceleration_structure_size)
                .usage(
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let blas_buffer = device
                .device
                .create_buffer(&blas_buffer_create_info, None)?;
            let blas_requirements = device.device.get_buffer_memory_requirements(blas_buffer);

            let blas_allocation = device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing BLAS",
                requirements: blas_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(blas_buffer),
            })?;

            device.device.bind_buffer_memory(
                blas_buffer,
                blas_allocation.memory(),
                blas_allocation.offset(),
            )?;

            let blas_create_info = vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(blas_buffer)
                .offset(0)
                .size(blas_size_info.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

            let blas = device
                .acceleration_structure_device
                .create_acceleration_structure(&blas_create_info, None)?;

            // Create scratch buffer for BLAS build
            let scratch_buffer_create_info = vk::BufferCreateInfo::default()
                .size(blas_size_info.build_scratch_size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let scratch_buffer = device
                .device
                .create_buffer(&scratch_buffer_create_info, None)?;
            let scratch_requirements = device.device.get_buffer_memory_requirements(scratch_buffer);

            let scratch_allocation = device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing Scratch Buffer",
                requirements: scratch_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(scratch_buffer),
            })?;

            device.device.bind_buffer_memory(
                scratch_buffer,
                scratch_allocation.memory(),
                scratch_allocation.offset(),
            )?;

            let scratch_address = device.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer),
            );

            // Build BLAS
            let blas_build_info = blas_build_info
                .dst_acceleration_structure(blas)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_address,
                });

            let blas_build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(primitive_count)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);

            // Create and submit command buffer for building BLAS
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(device.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);

            let command_pool = device
                .device
                .create_command_pool(&command_pool_create_info, None)?;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = device
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)?[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;

            device
                .acceleration_structure_device
                .cmd_build_acceleration_structures(
                    command_buffer,
                    &[blas_build_info],
                    &[&[blas_build_range]],
                );

            // Add memory barrier to ensure BLAS build completes
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);

            device.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );

            device.device.end_command_buffer(command_buffer)?;

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

            device
                .device
                .queue_submit(device.queue, &[submit_info], vk::Fence::null())?;
            device.device.queue_wait_idle(device.queue)?;

            device.device.destroy_command_pool(command_pool, None);
            device.device.destroy_buffer(scratch_buffer, None);
            device.free(scratch_allocation)?;

            // Get BLAS device address
            let blas_address = device
                .acceleration_structure_device
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(blas),
                );

            // Create instance buffer (3 cube instances along X with 1 unit gap between surfaces)
            // Cube is unit-sized ([-0.5, 0.5]), so center-to-center spacing is 2.0 for a 1.0 gap.
            let instance_x_offsets = [-2.0_f32, 0.0, 2.0];

            let instances: Vec<vk::AccelerationStructureInstanceKHR> = instance_x_offsets
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let transform_matrix = vk::TransformMatrixKHR {
                        // Row-major 3x4 affine transform. Translation lives in m03/m13/m23.
                        matrix: [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    };

                    vk::AccelerationStructureInstanceKHR {
                        transform: transform_matrix,
                        instance_custom_index_and_mask: vk::Packed24_8::new(i as u32, 0xFF),
                        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                            0,
                            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw()
                                as u8,
                        ),
                        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                            device_handle: blas_address,
                        },
                    }
                })
                .collect();

            let instance_count = instances.len() as u32;
            let instance_buffer_size =
                (instances.len() * size_of::<vk::AccelerationStructureInstanceKHR>()) as u64;
            let instance_buffer_create_info = vk::BufferCreateInfo::default()
                .size(instance_buffer_size)
                .usage(
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let instance_buffer = device
                .device
                .create_buffer(&instance_buffer_create_info, None)?;
            let instance_requirements = device
                .device
                .get_buffer_memory_requirements(instance_buffer);

            let mut instance_allocation = device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing Instance Buffer",
                requirements: instance_requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(instance_buffer),
            })?;

            device.device.bind_buffer_memory(
                instance_buffer,
                instance_allocation.memory(),
                instance_allocation.offset(),
            )?;

            // Copy instance data
            let instance_slice = instance_allocation
                .mapped_slice_mut()
                .ok_or_else(|| anyhow!("Instance buffer is not mapped"))?;

            // Safety: AccelerationStructureInstanceKHR is repr(C)
            let instance_bytes = std::slice::from_raw_parts(
                instances.as_ptr().cast::<u8>(),
                instance_buffer_size as usize,
            );
            instance_slice[..instance_buffer_size as usize].copy_from_slice(instance_bytes);

            let instance_buffer_address = device.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(instance_buffer),
            );

            // Build TLAS
            let tlas_geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                        .array_of_pointers(false)
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: instance_buffer_address,
                        }),
                });

            let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(&tlas_geometry));

            let mut tlas_size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            device
                .acceleration_structure_device
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &tlas_build_info,
                    &[instance_count],
                    &mut tlas_size_info,
                );

            // Create TLAS buffer
            let tlas_buffer_create_info = vk::BufferCreateInfo::default()
                .size(tlas_size_info.acceleration_structure_size)
                .usage(
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let tlas_buffer = device
                .device
                .create_buffer(&tlas_buffer_create_info, None)?;
            let tlas_requirements = device.device.get_buffer_memory_requirements(tlas_buffer);

            let tlas_allocation = device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing TLAS",
                requirements: tlas_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(tlas_buffer),
            })?;

            device.device.bind_buffer_memory(
                tlas_buffer,
                tlas_allocation.memory(),
                tlas_allocation.offset(),
            )?;

            let tlas_create_info = vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(tlas_buffer)
                .offset(0)
                .size(tlas_size_info.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

            let tlas = device
                .acceleration_structure_device
                .create_acceleration_structure(&tlas_create_info, None)?;

            // Create scratch buffer for TLAS build
            let scratch_buffer_create_info = vk::BufferCreateInfo::default()
                .size(tlas_size_info.build_scratch_size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let scratch_buffer = device
                .device
                .create_buffer(&scratch_buffer_create_info, None)?;
            let scratch_requirements = device.device.get_buffer_memory_requirements(scratch_buffer);

            let scratch_allocation = device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing Scratch Buffer",
                requirements: scratch_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(scratch_buffer),
            })?;

            device.device.bind_buffer_memory(
                scratch_buffer,
                scratch_allocation.memory(),
                scratch_allocation.offset(),
            )?;

            let scratch_address = device.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer),
            );

            // Build TLAS
            let tlas_build_info = tlas_build_info
                .dst_acceleration_structure(tlas)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_address,
                });

            let tlas_build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(instance_count)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);

            // Create and submit command buffer for building TLAS
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(device.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);

            let command_pool = device
                .device
                .create_command_pool(&command_pool_create_info, None)?;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = device
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)?[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;

            device
                .acceleration_structure_device
                .cmd_build_acceleration_structures(
                    command_buffer,
                    &[tlas_build_info],
                    &[&[tlas_build_range]],
                );

            device.device.end_command_buffer(command_buffer)?;

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

            device
                .device
                .queue_submit(device.queue, &[submit_info], vk::Fence::null())?;
            device.device.queue_wait_idle(device.queue)?;

            device.device.destroy_command_pool(command_pool, None);
            device.device.destroy_buffer(scratch_buffer, None);
            device.free(scratch_allocation)?;

            let pool_sizes = descriptor_set_layout_bindings
                .iter()
                .map(|binding| {
                    vk::DescriptorPoolSize::default()
                        .ty(binding.descriptor_type)
                        .descriptor_count(binding.descriptor_count)
                })
                .collect::<Vec<_>>();

            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);

            let descriptor_pool = device
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)?;

            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));

            let [descriptor_set] = device
                .device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)?
                .try_into()
                .map_err(|_| anyhow!("Failed to allocate exactly one descriptor set"))?;

            let image_info = vk::DescriptorImageInfo::default()
                .image_view(storage_image_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let mut accel_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(std::slice::from_ref(&tlas));

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&image_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .push_next(&mut accel_info),
            ];

            device
                .device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            Ok(Self {
                device,
                pipeline,
                pipeline_layout,
                shader_binding_table,
                shader_binding_table_allocation,
                sbt_raygen_region,
                sbt_miss_region,
                sbt_hit_region,
                sbt_callable_region,
                storage_image,
                storage_image_view,
                storage_image_allocation,
                storage_image_extent,
                vertex_buffer,
                vertex_allocation,
                index_buffer,
                index_allocation,
                blas,
                blas_buffer,
                blas_allocation,
                tlas,
                tlas_buffer,
                tlas_allocation,
                instance_buffer,
                instance_allocation,
                descriptor_set,
            })
        }
    }

    pub fn trace_rays(
        &self,
        command_buffer: vk::CommandBuffer,
        camera_transform: &Transform,
        camera_fov: f32,
        time_millis: u32,
    ) {
        self.device.transition_image(
            command_buffer,
            self.storage_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        let push_constants = PushConstants {
            viewport_width: self.storage_image_extent.width,
            viewport_height: self.storage_image_extent.height,
            camera_translation: camera_transform.translation,
            camera_rotation: Mat3::from_quat(camera_transform.rotation),
            camera_fov,
            time_millis,
        };

        unsafe {
            self.device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );

            self.device.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            self.device.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            let ray_tracing_pipeline_device = &self.device.ray_tracing_pipeline_device;
            ray_tracing_pipeline_device.cmd_trace_rays(
                command_buffer,
                &self.sbt_raygen_region,
                &self.sbt_miss_region,
                &self.sbt_hit_region,
                &self.sbt_callable_region,
                self.storage_image_extent.width,
                self.storage_image_extent.height,
                1,
            );
        }
    }

    pub fn blit(
        &self,
        command_buffer: vk::CommandBuffer,
        present_image: vk::Image,
        present_image_extent: vk::Extent2D,
    ) {
        self.device.transition_image(
            command_buffer,
            self.storage_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
        );

        let subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };

        let x = self.storage_image_extent.width as i32;
        let y = self.storage_image_extent.height as i32;

        let src_offsets = [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D { x, y, z: 1 },
        ];

        let x = present_image_extent.width as i32;
        let y = present_image_extent.height as i32;

        let dst_offsets = [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D { x, y, z: 1 },
        ];

        let image_blit = vk::ImageBlit::default()
            .src_subresource(subresource)
            .src_offsets(src_offsets)
            .dst_subresource(subresource)
            .dst_offsets(dst_offsets);

        unsafe {
            self.device.device.cmd_blit_image(
                command_buffer,
                self.storage_image,
                vk::ImageLayout::GENERAL,
                present_image,
                vk::ImageLayout::GENERAL,
                &[image_blit],
                vk::Filter::LINEAR,
            );
        }
    }

    pub fn recreate_storage_image(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            self.device.wait_idle()?;

            self.device
                .device
                .destroy_image_view(self.storage_image_view, None);

            self.device
                .free(std::mem::take(&mut self.storage_image_allocation))?;

            self.device.device.destroy_image(self.storage_image, None);

            let storage_image_extent = vk::Extent3D {
                width,
                height,
                depth: 1,
            };

            let storage_image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(storage_image_extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let storage_image = self
                .device
                .device
                .create_image(&storage_image_create_info, None)?;

            let requirements = self
                .device
                .device
                .get_image_memory_requirements(storage_image);

            let storage_image_allocation = self.device.allocate(&AllocationCreateDesc {
                name: "Ray Tracing Pipeline Storage Image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedImage(storage_image),
            })?;

            self.device.device.bind_image_memory(
                storage_image,
                storage_image_allocation.memory(),
                storage_image_allocation.offset(),
            )?;

            let storage_image_view_info = vk::ImageViewCreateInfo::default()
                .image(storage_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let storage_image_view = self
                .device
                .device
                .create_image_view(&storage_image_view_info, None)?;

            let image_info = vk::DescriptorImageInfo::default()
                .image_view(storage_image_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let write_descriptor_set = vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_info));

            self.device
                .device
                .update_descriptor_sets(&[write_descriptor_set], &[]);

            self.storage_image = storage_image;
            self.storage_image_view = storage_image_view;
            self.storage_image_allocation = storage_image_allocation;
            self.storage_image_extent = storage_image_extent;
        }

        Ok(())
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_image_view(self.storage_image_view, None);
            self.device.device.destroy_image(self.storage_image, None);
            self.device
                .free(std::mem::take(&mut self.storage_image_allocation))
                .unwrap();
            self.device
                .device
                .destroy_buffer(self.instance_buffer, None);
            self.device
                .free(std::mem::take(&mut self.instance_allocation))
                .unwrap();
            self.device
                .acceleration_structure_device
                .destroy_acceleration_structure(self.tlas, None);
            self.device.device.destroy_buffer(self.tlas_buffer, None);
            self.device
                .free(std::mem::take(&mut self.tlas_allocation))
                .unwrap();
            self.device
                .acceleration_structure_device
                .destroy_acceleration_structure(self.blas, None);
            self.device.device.destroy_buffer(self.blas_buffer, None);
            self.device
                .free(std::mem::take(&mut self.blas_allocation))
                .unwrap();
            self.device.device.destroy_buffer(self.index_buffer, None);
            self.device
                .free(std::mem::take(&mut self.index_allocation))
                .unwrap();
            self.device.device.destroy_buffer(self.vertex_buffer, None);
            self.device
                .free(std::mem::take(&mut self.vertex_allocation))
                .unwrap();
            self.device
                .device
                .destroy_buffer(self.shader_binding_table, None);
            self.device
                .free(std::mem::take(&mut self.shader_binding_table_allocation))
                .unwrap();
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct PushConstants {
    viewport_width: u32,
    viewport_height: u32,
    camera_translation: Vec3,
    camera_rotation: Mat3,
    camera_fov: f32,
    time_millis: u32,
}
