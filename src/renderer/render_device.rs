use std::{
    collections::HashSet,
    ffi::{CStr, c_char},
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow};
use ash::{khr, vk};
use bevy::{prelude::*, window::RawHandleWrapper};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};

use super::render_queue::RenderQueue;

#[derive(Resource, Clone, Deref)]
pub struct RenderDevice(Arc<RenderDeviceInner>);

#[expect(dead_code)]
pub struct RenderDeviceInner {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_instance: khr::surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device: ash::Device,
    pub swapchain_device: khr::swapchain::Device,
    pub ray_tracing_pipeline_device: khr::ray_tracing_pipeline::Device,
    pub acceleration_structure_device: khr::acceleration_structure::Device,
    pub deferred_host_operations_device: khr::deferred_host_operations::Device,
    pub allocator: Arc<Mutex<Allocator>>,
}

impl RenderDevice {
    pub fn new(handle: &RawHandleWrapper) -> Result<(Self, RenderQueue)> {
        unsafe {
            let entry = ash::Entry::load()?;
            let application_info = vk::ApplicationInfo::default().api_version(Self::api_version());
            let instance_layers = Self::instance_layers();
            let display_handle = handle.get_display_handle();
            let window_extensions = ash_window::enumerate_required_extensions(display_handle)?;

            let instance_create_info = vk::InstanceCreateInfo::default()
                .application_info(&application_info)
                .enabled_layer_names(&instance_layers)
                .enabled_extension_names(window_extensions);

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

            let device_memory_properties =
                instance.get_physical_device_memory_properties(physical_device);

            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&[1.0]);

            let device_extensions = Self::device_extensions();

            let device_extensions_properties =
                instance.enumerate_device_extension_properties(physical_device)?;

            let device_extension_properties = device_extensions_properties
                .iter()
                .map(|extension| extension.extension_name_as_c_str())
                .collect::<Result<HashSet<_>, _>>()?;

            let unsupported_device_extensions = device_extensions
                .iter()
                .filter(|extension| !device_extension_properties.contains(*extension))
                .collect::<Vec<_>>();

            if !unsupported_device_extensions.is_empty() {
                return Err(anyhow!(
                    "Unsupported device extensions: {:?}",
                    unsupported_device_extensions
                ));
            }

            let mut dynamic_rendering_features =
                vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

            let mut scalar_block_layout_features =
                vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true);

            let mut ray_tracing_pipeline_features =
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                    .ray_tracing_pipeline(true);

            let mut acceleration_structure_features =
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                    .acceleration_structure(true);

            let mut buffer_device_address_features =
                vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
                    .buffer_device_address(true);

            let mut synchronization2_features =
                vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

            let device_extensions = device_extensions
                .into_iter()
                .map(|extension| extension.as_ptr())
                .collect::<Vec<_>>();

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&device_extensions)
                .push_next(&mut dynamic_rendering_features)
                .push_next(&mut scalar_block_layout_features)
                .push_next(&mut ray_tracing_pipeline_features)
                .push_next(&mut acceleration_structure_features)
                .push_next(&mut buffer_device_address_features)
                .push_next(&mut synchronization2_features);

            let device = instance.create_device(physical_device, &device_create_info, None)?;
            let swapchain_device = khr::swapchain::Device::new(&instance, &device);
            let ray_tracing_pipeline_device =
                khr::ray_tracing_pipeline::Device::new(&instance, &device);
            let acceleration_structure_device =
                khr::acceleration_structure::Device::new(&instance, &device);
            let deferred_host_operations_device =
                khr::deferred_host_operations::Device::new(&instance, &device);

            let allocater_create_desc = AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: true,
                allocation_sizes: Default::default(),
            };

            let allocator = Arc::new(Mutex::new(Allocator::new(&allocater_create_desc)?));

            let render_device = Self(Arc::new(RenderDeviceInner {
                entry,
                instance,
                surface_instance,
                physical_device,
                device_memory_properties,
                device,
                swapchain_device,
                ray_tracing_pipeline_device,
                acceleration_structure_device,
                deferred_host_operations_device,
                allocator,
            }));

            let render_queue = RenderQueue::new(render_device.clone(), queue_family_index, 0);
            Ok((render_device, render_queue))
        }
    }

    pub fn allocate(&self, desc: &AllocationCreateDesc) -> Allocation {
        let mut allocator = self.allocator.lock().unwrap();
        let allocation = allocator.allocate(desc).expect("Failed to allocate memory");
        allocation
    }

    pub fn free(&self, allocation: Allocation) {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.free(allocation).expect("Failed to free memory");
    }

    pub fn wait_idle(&self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed while waiting for device to become idle");
        }
    }

    pub fn get_physical_device_ray_tracing_pipeline_properties(
        &self,
    ) -> vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'_> {
        unsafe {
            let mut props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props);
            self.instance
                .get_physical_device_properties2(self.physical_device, &mut props2);
            props
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

    fn device_extensions() -> Vec<&'static CStr> {
        vec![
            khr::swapchain::NAME,
            khr::dynamic_rendering::NAME,
            khr::ray_tracing_pipeline::NAME,
            khr::acceleration_structure::NAME,
            khr::deferred_host_operations::NAME,
        ]
    }
}
