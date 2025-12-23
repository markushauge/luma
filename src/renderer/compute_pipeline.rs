use anyhow::{Result, anyhow};
use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::shader::Shader;

use super::{Device, Frame};

#[allow(dead_code)]
pub struct ComputePipeline {
    pub device: Device,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout_bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl ComputePipeline {
    pub fn new(device: Device, shader: &Shader) -> Result<Self> {
        unsafe {
            let shader_module_create_info =
                vk::ShaderModuleCreateInfo::default().code(&shader.code);

            let compute_shader_module = device
                .device
                .create_shader_module(&shader_module_create_info, None)?;

            let descriptor_set_layout_bindings = vec![
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];

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

            Ok(Self {
                device,
                pipeline,
                pipeline_layout,
                descriptor_set_layout_bindings,
                descriptor_set_layout,
            })
        }
    }

    pub fn dispatch(&self, frame: &Frame, camera_transform: &Transform, time_millis: u32) {
        self.device.transition_image(
            frame.command_buffer,
            frame.storage_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        let push_constants = PushConstants {
            viewport_width: frame.storage_image_extent.width,
            viewport_height: frame.storage_image_extent.height,
            camera_translation: camera_transform.translation,
            camera_rotation: Mat3::from_quat(camera_transform.rotation),
            camera_fov: 52.0f32.to_radians(), // TODO: Make configurable
            time_millis,
        };

        unsafe {
            self.device.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            self.device.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[frame.descriptor_set],
                &[],
            );

            self.device.device.cmd_push_constants(
                frame.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            self.device.device.cmd_dispatch(
                frame.command_buffer,
                frame.storage_image_extent.width.div_ceil(16),
                frame.storage_image_extent.height.div_ceil(16),
                1,
            );
        }
    }

    pub fn blit(
        &self,
        frame: &Frame,
        present_image: vk::Image,
        present_image_extent: vk::Extent2D,
    ) {
        self.device.transition_image(
            frame.command_buffer,
            frame.storage_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
        );

        let subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };

        let x = frame.storage_image_extent.width as i32;
        let y = frame.storage_image_extent.height as i32;

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
                frame.command_buffer,
                frame.storage_image,
                vk::ImageLayout::GENERAL,
                present_image,
                vk::ImageLayout::GENERAL,
                &[image_blit],
                vk::Filter::LINEAR,
            );
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
