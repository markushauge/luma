use std::{ffi::c_char, sync::Arc};

use anyhow::{Result, anyhow};
use ash::{khr, vk};
use bevy::{prelude::*, window::RawHandleWrapper};

use super::Frame;

#[derive(Clone, Deref)]
pub struct Device(Arc<DeviceInner>);

#[allow(dead_code)]
pub struct DeviceInner {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_instance: khr::surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub device: ash::Device,
    pub swapchain_device: khr::swapchain::Device,
    pub queue: vk::Queue,
}

impl Device {
    pub fn new(raw_handles: &RawHandleWrapper) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load()?;
            let application_info = vk::ApplicationInfo::default().api_version(Self::api_version());
            let instance_layers = Self::instance_layers();
            let mut instance_extensions = Self::instance_extensions();
            let display_handle = raw_handles.get_display_handle();
            let window_extensions = ash_window::enumerate_required_extensions(display_handle)?;
            instance_extensions.extend(window_extensions);
            let instance_create_flags = Self::instance_create_flags();

            let instance_create_info = vk::InstanceCreateInfo::default()
                .application_info(&application_info)
                .enabled_layer_names(&instance_layers)
                .enabled_extension_names(&instance_extensions)
                .flags(instance_create_flags);

            let instance = entry.create_instance(&instance_create_info, None)?;
            let surface_instance = khr::surface::Instance::new(&entry, &instance);

            let (physical_device, queue_family_index) = instance
                .enumerate_physical_devices()?
                .into_iter()
                .find_map(|physical_device| {
                    instance
                        .get_physical_device_queue_family_properties(physical_device)
                        .into_iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            info.queue_flags
                                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                                .then_some((physical_device, index as u32))
                        })
                })
                .ok_or_else(|| anyhow!("No suitable physical device found"))?;

            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&[1.0]);

            let device_extensions = Self::device_extensions();

            let mut dynamic_rendering_features =
                vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

            let mut scalar_block_layout_features =
                vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&device_extensions)
                .push_next(&mut dynamic_rendering_features)
                .push_next(&mut scalar_block_layout_features);

            let device = instance.create_device(physical_device, &device_create_info, None)?;
            let swapchain_device = khr::swapchain::Device::new(&instance, &device);
            let queue = device.get_device_queue(queue_family_index, 0);

            let inner = DeviceInner {
                entry,
                instance,
                surface_instance,
                physical_device,
                queue_family_index,
                device,
                swapchain_device,
                queue,
            };

            Ok(Self(Arc::new(inner)))
        }
    }

    pub fn begin_frame(&self, frame: &Frame) -> Result<()> {
        unsafe {
            self.device
                .wait_for_fences(&[frame.fence], true, u64::MAX)?;

            self.device.reset_fences(&[frame.fence])?;

            self.device.reset_command_buffer(
                frame.command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )?;

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(frame.command_buffer, &command_buffer_begin_info)?;

            Ok(())
        }
    }

    pub fn end_frame(&self, frame: &Frame) -> Result<()> {
        unsafe {
            self.device.end_command_buffer(frame.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(std::slice::from_ref(&frame.present_complete_semaphore))
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])
                .command_buffers(std::slice::from_ref(&frame.command_buffer))
                .signal_semaphores(std::slice::from_ref(&frame.rendering_complete_semaphore));

            self.device
                .queue_submit(self.queue, &[submit_info], frame.fence)?;

            Ok(())
        }
    }

    pub fn transition_image(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(subresource_range)
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ);

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    fn api_version() -> u32 {
        vk::API_VERSION_1_3
    }

    fn instance_layers() -> Vec<*const c_char> {
        let mut instance_layers = vec![];

        // Enable validation layers in debug mode
        if cfg!(debug_assertions) {
            instance_layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }

        instance_layers
    }

    fn instance_extensions() -> Vec<*const c_char> {
        let mut instance_extensions = vec![];

        // Enable portability enumeration on macOS
        if cfg!(target_os = "macos") {
            instance_extensions.push(khr::portability_enumeration::NAME.as_ptr());
        }

        instance_extensions
    }

    fn instance_create_flags() -> vk::InstanceCreateFlags {
        let mut instance_create_flags = vk::InstanceCreateFlags::empty();

        // Enable portability enumeration on macOS
        if cfg!(target_os = "macos") {
            instance_create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        instance_create_flags
    }

    fn device_extensions() -> Vec<*const c_char> {
        let mut device_extensions = vec![
            khr::swapchain::NAME.as_ptr(),
            khr::dynamic_rendering::NAME.as_ptr(),
        ];

        // Enable portability subset on macOS
        if cfg!(target_os = "macos") {
            device_extensions.push(khr::portability_subset::NAME.as_ptr());
        }

        device_extensions
    }
}
