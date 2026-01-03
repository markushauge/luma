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

pub struct ComputePipelinePlugin {
    pub settings: ComputePipelineSettings,
}

impl Plugin for ComputePipelinePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.settings.clone())
            .add_systems(Startup, load_shader)
            .add_systems(
                Render,
                (create_compute_pipeline, run_compute_pipeline).in_set(RenderSystems::Render),
            );
    }
}

#[derive(Resource, Clone)]
pub struct ComputePipelineSettings {
    pub resolution_scaling: f32,
}

impl Default for ComputePipelineSettings {
    fn default() -> Self {
        Self {
            resolution_scaling: 1.0,
        }
    }
}

#[derive(Resource)]
struct ComputeShader(Handle<Shader>);

fn load_shader(mut commands: Commands, asset_server: Res<AssetServer>) {
    let shader = asset_server.load("shaders/main.comp");
    commands.insert_resource(ComputeShader(shader));
}

fn create_compute_pipeline(
    mut commands: Commands,
    renderer: Res<Renderer>,
    compute_pipeline: Option<Res<ComputePipeline>>,
    settings: Res<ComputePipelineSettings>,
    compute_shader: Res<ComputeShader>,
    assets: Res<Assets<Shader>>,
) -> Result<(), BevyError> {
    if compute_pipeline.is_some() {
        return Ok(());
    }

    let Some(shader) = assets.get(&compute_shader.0) else {
        return Ok(());
    };

    tracing::info!("Creating compute pipeline");

    let Extent2D { width, height } = renderer.swapchain.surface_extent;
    let width = (width as f32 * settings.resolution_scaling) as u32;
    let height = (height as f32 * settings.resolution_scaling) as u32;

    let compute_pipeline = ComputePipeline::new(renderer.device.clone(), shader, width, height)?;
    commands.insert_resource(compute_pipeline);

    Ok(())
}

fn run_compute_pipeline(
    renderer: Res<Renderer>,
    compute_pipeline: Option<Res<ComputePipeline>>,
    camera_transform: Query<&Transform, With<Camera>>,
    time: Res<Time>,
) -> Result<(), BevyError> {
    let Some(compute_pipeline) = compute_pipeline else {
        return Ok(());
    };

    let camera_transform = camera_transform.single()?;

    compute_pipeline.dispatch(
        renderer.command_buffer,
        camera_transform,
        time.elapsed().as_millis() as u32,
    );

    compute_pipeline.blit(
        renderer.command_buffer,
        renderer.swapchain.present_image().image,
        renderer.swapchain.surface_extent,
    );

    Ok(())
}

#[derive(Resource)]
pub struct ComputePipeline {
    pub device: Device,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub storage_image: vk::Image,
    pub storage_image_allocation: Allocation,
    pub storage_image_extent: vk::Extent3D,
    pub descriptor_set: vk::DescriptorSet,
}

impl ComputePipeline {
    pub fn new(device: Device, shader: &Shader, width: u32, height: u32) -> Result<Self> {
        unsafe {
            let shader_module_create_info =
                vk::ShaderModuleCreateInfo::default().code(&shader.code);

            let compute_shader_module = device
                .device
                .create_shader_module(&shader_module_create_info, None)?;

            let descriptor_set_layout_bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];

            let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&descriptor_set_layout_bindings);

            let descriptor_set_layout = device
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?;

            let push_constant_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(size_of::<PushConstants>() as u32);

            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(std::slice::from_ref(&push_constant_range));

            let pipeline_layout = device
                .device
                .create_pipeline_layout(&pipeline_layout_create_info, None)?;

            let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(compute_shader_module)
                .name(&shader.entry_point);

            let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
                .stage(shader_stage_create_info)
                .layout(pipeline_layout);

            let [pipeline] = device
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .map_err(|(_, result)| anyhow!("Failed to create compute pipeline: {:?}", result))?
                .try_into()
                .map_err(|_| anyhow!("Failed to create exactly one compute pipeline"))?;

            device
                .device
                .destroy_shader_module(compute_shader_module, None);

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
                name: "Compute Pipeline Storage Image",
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

            let write_descriptor_set = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_info));

            device
                .device
                .update_descriptor_sets(&[write_descriptor_set], &[]);

            Ok(Self {
                device,
                pipeline,
                pipeline_layout,
                storage_image,
                storage_image_allocation,
                storage_image_extent,
                descriptor_set,
            })
        }
    }

    pub fn dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        camera_transform: &Transform,
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
            camera_fov: 52.0f32.to_radians(), // TODO: Make configurable
            time_millis,
        };

        unsafe {
            self.device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            self.device.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            self.device.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            self.device.device.cmd_dispatch(
                command_buffer,
                self.storage_image_extent.width.div_ceil(16),
                self.storage_image_extent.height.div_ceil(16),
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
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        self.device
            .free(std::mem::take(&mut self.storage_image_allocation))
            .unwrap();
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
