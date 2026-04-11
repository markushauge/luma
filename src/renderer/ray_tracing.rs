use anyhow::{Result, anyhow};
use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::{camera::Camera, shader::Shader};

use super::{
    RenderDevice, Renderer,
    acceleration_structure::{AccelerationStructureManager, AccelerationStructurePlugin, Tlas},
    buffer::Buffer,
    resource_state_tracker::{ImageState, ResourceStateTracker},
    schedule::{Render, RenderSystems},
    storage_image::StorageImage,
};

#[derive(Default)]
pub struct RayTracingPlugin {
    pub settings: RayTracingSettings,
}

impl Plugin for RayTracingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.settings.clone())
            .add_plugins(AccelerationStructurePlugin)
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
    mut renderer: ResMut<Renderer>,
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

    let Some(closest_hit_shader) = assets.get(&ray_tracing_shaders.closest_hit) else {
        return Ok(());
    };

    let vk::Extent2D { width, height } = renderer.swapchain.surface_extent;

    let extent = vk::Extent2D {
        width: (width as f32 * settings.resolution_scaling) as u32,
        height: (height as f32 * settings.resolution_scaling) as u32,
    };

    match ray_tracing_pipeline {
        None => {
            commands.insert_resource(
                RayTracingPipeline::builder(renderer.render_device.clone())
                    .with_raygen_shader_group(raygen_shader)?
                    .with_miss_shader_group(miss_shader)?
                    .with_hit_shader_group(closest_hit_shader, None, None)?
                    .build(extent)?,
            );
        }
        Some(mut ray_tracing_pipeline) => {
            if ray_tracing_pipeline.storage_image.extent != extent {
                ray_tracing_pipeline.recreate_storage_image(extent, &mut renderer.tracker)?;
            }
        }
    }

    Ok(())
}

fn execute_ray_tracing_pipeline(
    mut renderer: ResMut<Renderer>,
    ray_tracing_pipeline: Option<Res<RayTracingPipeline>>,
    acceleration_structure_manager: Res<AccelerationStructureManager>,
    camera: Query<(&Camera, &Transform), With<Camera>>,
    time: Res<Time>,
) -> Result<(), BevyError> {
    let Some(ray_tracing_pipeline) = ray_tracing_pipeline else {
        return Ok(());
    };

    let Some(tlas) = acceleration_structure_manager.tlas() else {
        return Ok(());
    };

    let (camera, camera_transform) = camera.single()?;

    let command_buffer = renderer.render_context.command_buffer;
    let swapchain_image = renderer.swapchain.current_image().image;
    let swapchain_image_extent = renderer.swapchain.surface_extent;
    let tracker = &mut renderer.tracker;

    ray_tracing_pipeline.trace_rays(
        command_buffer,
        tracker,
        tlas,
        camera_transform,
        camera.vertical_fov(),
        time.elapsed().as_millis() as u32,
    );

    ray_tracing_pipeline.blit(
        command_buffer,
        tracker,
        swapchain_image,
        swapchain_image_extent,
    );

    Ok(())
}

#[derive(Resource)]
pub struct RayTracingPipeline {
    pub render_device: RenderDevice,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub shader_binding_table: ShaderBindingTable,
    pub storage_image: StorageImage,
    pub descriptor_set: vk::DescriptorSet,
}

impl RayTracingPipeline {
    pub fn builder<'a>(render_device: RenderDevice) -> RayTracingPipelineBuilder<'a> {
        RayTracingPipelineBuilder::new(render_device)
    }

    pub fn trace_rays(
        &self,
        command_buffer: vk::CommandBuffer,
        tracker: &mut ResourceStateTracker,
        tlas: &Tlas,
        camera_transform: &Transform,
        camera_fov: f32,
        time_millis: u32,
    ) {
        unsafe {
            let mut acceleration_structure_info =
                vk::WriteDescriptorSetAccelerationStructureKHR::default()
                    .acceleration_structures(std::slice::from_ref(tlas.acceleration_structure()));

            self.render_device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .push_next(&mut acceleration_structure_info)],
                &[],
            );
        }

        tracker
            .transition_image(
                self.storage_image.image,
                ImageState {
                    layout: vk::ImageLayout::GENERAL,
                    access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    stages: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                },
            )
            .flush(&self.render_device, command_buffer);

        let push_constants = PushConstants {
            viewport_width: self.storage_image.extent.width,
            viewport_height: self.storage_image.extent.height,
            camera_translation: camera_transform.translation,
            camera_rotation: Mat3::from_quat(camera_transform.rotation),
            camera_fov,
            time_millis,
        };

        unsafe {
            self.render_device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );

            self.render_device.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            self.render_device.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            self.render_device
                .ray_tracing_pipeline_device
                .cmd_trace_rays(
                    command_buffer,
                    &self.shader_binding_table.raygen_region,
                    &self.shader_binding_table.miss_region,
                    &self.shader_binding_table.hit_region,
                    &self.shader_binding_table.callable_region,
                    self.storage_image.extent.width,
                    self.storage_image.extent.height,
                    1,
                );
        }
    }

    pub fn blit(
        &self,
        command_buffer: vk::CommandBuffer,
        tracker: &mut ResourceStateTracker,
        swapchain_image: vk::Image,
        swapchain_image_extent: vk::Extent2D,
    ) {
        tracker
            .transition_image(
                self.storage_image.image,
                ImageState {
                    layout: vk::ImageLayout::GENERAL,
                    access: vk::AccessFlags2::TRANSFER_READ,
                    stages: vk::PipelineStageFlags2::TRANSFER,
                },
            )
            .flush(&self.render_device, command_buffer);

        let subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };

        let x = self.storage_image.extent.width as i32;
        let y = self.storage_image.extent.height as i32;

        let src_offsets = [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D { x, y, z: 1 },
        ];

        let x = swapchain_image_extent.width as i32;
        let y = swapchain_image_extent.height as i32;

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
            self.render_device.device.cmd_blit_image(
                command_buffer,
                self.storage_image.image,
                vk::ImageLayout::GENERAL,
                swapchain_image,
                vk::ImageLayout::GENERAL,
                &[image_blit],
                vk::Filter::LINEAR,
            );
        }
    }

    pub fn recreate_storage_image(
        &mut self,
        extent: vk::Extent2D,
        tracker: &mut ResourceStateTracker,
    ) -> Result<()> {
        self.render_device.wait_idle()?;

        let new_storage_image = self.render_device.create_storage_image(
            extent,
            vk::Format::R8G8B8A8_UNORM,
            Some("Ray Tracing Pipeline Storage Image"),
        )?;

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(new_storage_image.image_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let write_descriptor_set = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_info));

        unsafe {
            self.render_device
                .device
                .update_descriptor_sets(&[write_descriptor_set], &[]);
        }

        let old_storage_image = std::mem::replace(&mut self.storage_image, new_storage_image);
        tracker.untrack_image(old_storage_image.image);
        self.render_device.destroy_storage_image(old_storage_image);
        Ok(())
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.shader_binding_table.destroy(&self.render_device);
            self.render_device
                .destroy_storage_image(std::mem::take(&mut self.storage_image));
            self.render_device
                .device
                .destroy_pipeline(self.pipeline, None);
            self.render_device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

pub struct ShaderBindingTable {
    buffer: Buffer,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
    callable_region: vk::StridedDeviceAddressRegionKHR,
}

impl ShaderBindingTable {
    unsafe fn destroy(&mut self, render_device: &RenderDevice) {
        render_device.destroy_buffer(std::mem::take(&mut self.buffer));
    }
}

pub struct RayTracingPipelineBuilder<'a> {
    render_device: RenderDevice,
    shader_modules: Vec<vk::ShaderModule>,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    shader_groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR<'a>>,
    raygen_group_indices: Vec<usize>,
    miss_group_indices: Vec<usize>,
    hit_group_indices: Vec<usize>,
}

impl<'a> RayTracingPipelineBuilder<'a> {
    pub fn new(render_device: RenderDevice) -> Self {
        Self {
            render_device,
            shader_modules: Vec::new(),
            shader_stages: Vec::new(),
            shader_groups: Vec::new(),
            raygen_group_indices: Vec::new(),
            miss_group_indices: Vec::new(),
            hit_group_indices: Vec::new(),
        }
    }

    pub fn with_raygen_shader_group(mut self, shader: &'a Shader) -> Result<Self, vk::Result> {
        let raygen_stage_index =
            self.push_shader_stage(shader, vk::ShaderStageFlags::RAYGEN_KHR)?;

        let raygen_group_index = self.push_shader_group(
            vk::RayTracingShaderGroupTypeKHR::GENERAL,
            Some(raygen_stage_index),
            None,
            None,
            None,
        );

        self.raygen_group_indices.push(raygen_group_index);
        Ok(self)
    }

    pub fn with_miss_shader_group(mut self, shader: &'a Shader) -> Result<Self, vk::Result> {
        let miss_stage_index = self.push_shader_stage(shader, vk::ShaderStageFlags::MISS_KHR)?;

        let group_index = self.push_shader_group(
            vk::RayTracingShaderGroupTypeKHR::GENERAL,
            Some(miss_stage_index),
            None,
            None,
            None,
        );

        self.miss_group_indices.push(group_index);
        Ok(self)
    }

    pub fn with_hit_shader_group(
        mut self,
        closest_hit_shader: &'a Shader,
        any_hit_shader: Option<&'a Shader>,
        intersection_shader: Option<&'a Shader>,
    ) -> Result<Self, vk::Result> {
        let closest_hit_stage_index =
            self.push_shader_stage(closest_hit_shader, vk::ShaderStageFlags::CLOSEST_HIT_KHR)?;

        let any_hit_stage_index = any_hit_shader
            .map(|any_hit_shader| {
                self.push_shader_stage(any_hit_shader, vk::ShaderStageFlags::ANY_HIT_KHR)
            })
            .transpose()?;

        let intersection_stage_index = intersection_shader
            .map(|intersection_shader| {
                self.push_shader_stage(intersection_shader, vk::ShaderStageFlags::INTERSECTION_KHR)
            })
            .transpose()?;

        let shader_group_type = if intersection_shader.is_some() {
            vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
        } else {
            vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
        };

        let group_index = self.push_shader_group(
            shader_group_type,
            None,
            Some(closest_hit_stage_index),
            any_hit_stage_index,
            intersection_stage_index,
        );

        self.hit_group_indices.push(group_index);
        Ok(self)
    }

    pub fn build(self, extent: vk::Extent2D) -> Result<RayTracingPipeline> {
        unsafe {
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

            let descriptor_set_layout = self
                .render_device
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?;

            let push_constant_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                .offset(0)
                .size(size_of::<PushConstants>() as u32);

            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(std::slice::from_ref(&push_constant_range));

            let pipeline_layout = self
                .render_device
                .device
                .create_pipeline_layout(&pipeline_layout_create_info, None)?;

            let pipeline_create_info = vk::RayTracingPipelineCreateInfoKHR::default()
                .stages(&self.shader_stages)
                .groups(&self.shader_groups)
                .max_pipeline_ray_recursion_depth(1)
                .layout(pipeline_layout);

            let ray_tracing_pipeline_device = &self.render_device.ray_tracing_pipeline_device;

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

            let shader_binding_table = self.build_sbt(pipeline)?;

            for shader_module in self.shader_modules {
                self.render_device
                    .device
                    .destroy_shader_module(shader_module, None);
            }

            let storage_image = self.render_device.create_storage_image(
                extent,
                vk::Format::R8G8B8A8_UNORM,
                Some("Ray Tracing Pipeline Storage Image"),
            )?;

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

            let descriptor_pool = self
                .render_device
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)?;

            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));

            let [descriptor_set] = self
                .render_device
                .device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)?
                .try_into()
                .map_err(|_| anyhow!("Failed to allocate exactly one descriptor set"))?;

            let image_info = vk::DescriptorImageInfo::default()
                .image_view(storage_image.image_view)
                .image_layout(vk::ImageLayout::GENERAL);

            self.render_device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&image_info))],
                &[],
            );

            Ok(RayTracingPipeline {
                render_device: self.render_device,
                pipeline,
                pipeline_layout,
                shader_binding_table,
                storage_image,
                descriptor_set,
            })
        }
    }

    fn push_shader_stage(
        &mut self,
        shader: &'a Shader,
        stage_flag: vk::ShaderStageFlags,
    ) -> Result<u32, vk::Result> {
        let shader_module = unsafe {
            self.render_device.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&shader.code),
                None,
            )?
        };

        self.shader_modules.push(shader_module);

        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(stage_flag)
            .module(shader_module)
            .name(&shader.entry_point);

        let stage_index = self.shader_stages.len() as u32;
        self.shader_stages.push(shader_stage);
        Ok(stage_index)
    }

    fn push_shader_group(
        &mut self,
        shader_group_type: vk::RayTracingShaderGroupTypeKHR,
        general_shader_stage_index: Option<u32>,
        closest_hit_shader_stage_index: Option<u32>,
        any_hit_shader_stage_index: Option<u32>,
        intersection_shader_stage_index: Option<u32>,
    ) -> usize {
        let shader_group = vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(shader_group_type)
            .general_shader(general_shader_stage_index.unwrap_or(vk::SHADER_UNUSED_KHR))
            .closest_hit_shader(closest_hit_shader_stage_index.unwrap_or(vk::SHADER_UNUSED_KHR))
            .any_hit_shader(any_hit_shader_stage_index.unwrap_or(vk::SHADER_UNUSED_KHR))
            .intersection_shader(intersection_shader_stage_index.unwrap_or(vk::SHADER_UNUSED_KHR));

        let group_index = self.shader_groups.len();
        self.shader_groups.push(shader_group);
        group_index
    }

    fn build_sbt(&self, pipeline: vk::Pipeline) -> Result<ShaderBindingTable> {
        let rt_props = self
            .render_device
            .get_physical_device_ray_tracing_pipeline_properties();

        let handle_size = rt_props.shader_group_handle_size as usize;
        let handle_alignment = rt_props.shader_group_handle_alignment as usize;
        let base_alignment = rt_props.shader_group_base_alignment as usize;

        let region_stride = handle_size.next_multiple_of(handle_alignment);

        // Each region is packed as count * region_stride, aligned to base_alignment.
        // Raygen: the spec requires stride == size (only one raygen is dispatched).
        let raygen_region_size = region_stride.next_multiple_of(base_alignment);
        let miss_region_size =
            (self.miss_group_indices.len() * region_stride).next_multiple_of(base_alignment);
        let hit_region_size =
            (self.hit_group_indices.len() * region_stride).next_multiple_of(base_alignment);

        let raygen_region_offset = 0usize;
        let miss_region_offset = raygen_region_offset + raygen_region_size;
        let hit_region_offset = miss_region_offset + miss_region_size;

        let sbt_size = hit_region_offset + hit_region_size;

        let mut buffer = self.render_device.create_buffer(
            sbt_size as u64,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
            Some("Shader Binding Table Buffer"),
        )?;

        let group_count = self.shader_groups.len();
        let shader_handles = unsafe {
            self.render_device
                .ray_tracing_pipeline_device
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count as u32,
                    handle_size * group_count,
                )
                .map_err(|_| anyhow!("Failed to get shader group handles"))?
        };

        let sbt_data = buffer.slice_mut()?;
        sbt_data.fill(0);

        for (slot, &group_index) in self.raygen_group_indices.iter().enumerate() {
            let dst = raygen_region_offset + slot * region_stride;
            let src = group_index * handle_size;
            sbt_data[dst..][..handle_size].copy_from_slice(&shader_handles[src..][..handle_size]);
        }

        for (slot, &group_index) in self.miss_group_indices.iter().enumerate() {
            let dst = miss_region_offset + slot * region_stride;
            let src = group_index * handle_size;
            sbt_data[dst..][..handle_size].copy_from_slice(&shader_handles[src..][..handle_size]);
        }

        for (slot, &group_index) in self.hit_group_indices.iter().enumerate() {
            let dst = hit_region_offset + slot * region_stride;
            let src = group_index * handle_size;
            sbt_data[dst..][..handle_size].copy_from_slice(&shader_handles[src..][..handle_size]);
        }

        let sbt_address = self.render_device.get_buffer_device_address(&buffer);

        let raygen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + raygen_region_offset as u64)
            .stride(raygen_region_size as u64)
            .size(raygen_region_size as u64);

        let miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + miss_region_offset as u64)
            .stride(region_stride as u64)
            .size(miss_region_size as u64);

        let hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + hit_region_offset as u64)
            .stride(region_stride as u64)
            .size(hit_region_size as u64);

        let callable_region = vk::StridedDeviceAddressRegionKHR::default();

        Ok(ShaderBindingTable {
            buffer,
            raygen_region,
            miss_region,
            hit_region,
            callable_region,
        })
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
