use std::ffi::CStr;

use anyhow::Result;
use ash::{khr, vk};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowAttributes, WindowId},
};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let window_attributes = WindowAttributes::default().with_title("Lys");
    let mut app = App::new(window_attributes);
    let event_loop = EventLoop::builder().build()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    window_attributes: WindowAttributes,
    window_state: Option<WindowState>,
}

struct WindowState {
    window: Window,
    device: Device,
}

impl App {
    fn new(window_attributes: WindowAttributes) -> Self {
        Self {
            window_attributes,
            window_state: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        tracing::info!("Application resumed");

        if self.window_state.is_some() {
            return;
        }

        let window = match event_loop.create_window(self.window_attributes.clone()) {
            Ok(window) => window,
            Err(err) => {
                tracing::error!("Failed to create window: {:?}", err);
                return;
            }
        };

        let device = match Device::new(&window) {
            Ok(device) => device,
            Err(err) => {
                tracing::error!("Failed to create device: {:?}", err);
                return;
            }
        };

        window.request_redraw();
        self.window_state = Some(WindowState { window, device });
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(WindowState { window, device }) = self.window_state.as_ref() else {
            return;
        };

        if let WindowEvent::RedrawRequested = event {
            if let Err(err) = device.render() {
                tracing::error!("Failed to render: {:?}", err);
            }

            window.request_redraw();
        }
    }
}

#[allow(dead_code)]
struct Device {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue_family_index: u32,
    physical_device: vk::PhysicalDevice,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_device: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    present_queue: vk::Queue,
    present_images: Vec<vk::Image>,
    present_complete_semaphore: vk::Semaphore,
    rendering_complete_semaphore: vk::Semaphore,
    command_buffer: vk::CommandBuffer,
    command_buffer_fence: vk::Fence,
    pool: vk::CommandPool,
}

impl Device {
    fn new(window: &Window) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load()?;

            let instance_layers = [c"VK_LAYER_KHRONOS_validation"]
                .into_iter()
                .map(CStr::as_ptr)
                .collect::<Vec<_>>();

            let mut instance_extensions = [
                khr::portability_enumeration::NAME,
                khr::get_physical_device_properties2::NAME,
            ]
            .into_iter()
            .map(CStr::as_ptr)
            .collect::<Vec<_>>();

            let window_handle = window.window_handle()?.as_raw();
            let display_handle = window.display_handle()?.as_raw();
            let window_extensions = ash_window::enumerate_required_extensions(display_handle)?;
            instance_extensions.extend(window_extensions);

            let application_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

            let instance_create_info = vk::InstanceCreateInfo::default()
                .application_info(&application_info)
                .enabled_layer_names(&instance_layers)
                .enabled_extension_names(&instance_extensions)
                .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

            let instance = entry.create_instance(&instance_create_info, None)?;

            let surface =
                ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?;

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
                            if !info
                                .queue_flags
                                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                            {
                                return None;
                            }

                            if !surface_instance
                                .get_physical_device_surface_support(
                                    physical_device,
                                    index as u32,
                                    surface,
                                )
                                .ok()?
                            {
                                return None;
                            }

                            Some((physical_device, index as u32))
                        })
                })
                .ok_or_else(|| anyhow::anyhow!("No suitable physical device found"))?;

            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&[1.0]);

            let device_extensions = [
                khr::swapchain::NAME,
                khr::portability_subset::NAME,
                khr::dynamic_rendering::NAME,
            ]
            .into_iter()
            .map(CStr::as_ptr)
            .collect::<Vec<_>>();

            // Enable dynamic rendering feature
            let mut dynamic_rendering_features =
                vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&device_extensions)
                .push_next(&mut dynamic_rendering_features);

            let device = instance.create_device(physical_device, &device_create_info, None)?;
            let present_queue = device.get_device_queue(queue_family_index, 0);

            let surface_formats =
                surface_instance.get_physical_device_surface_formats(physical_device, surface)?;

            let surface_format = surface_formats
                .iter()
                .find(|format| {
                    format.format == vk::Format::B8G8R8A8_SRGB
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .or_else(|| surface_formats.first())
                .ok_or_else(|| anyhow::anyhow!("No suitable surface format found"))?;

            tracing::info!("Using surface format: {:?}", surface_format);

            let surface_capabilities = surface_instance
                .get_physical_device_surface_capabilities(physical_device, surface)?;

            let mut desired_image_count = surface_capabilities.min_image_count + 1;

            if surface_capabilities.max_image_count > 0 {
                desired_image_count = desired_image_count.min(surface_capabilities.max_image_count);
            }

            let window_size = window.inner_size();

            let surface_resolution = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: window_size.width,
                    height: window_size.height,
                },
                _ => surface_capabilities.current_extent,
            };

            let present_mode = surface_instance
                .get_physical_device_surface_present_modes(physical_device, surface)?
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_device = khr::swapchain::Device::new(&instance, &device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(surface_resolution)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(std::slice::from_ref(&queue_family_index))
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);

            let swapchain = swapchain_device.create_swapchain(&swapchain_create_info, None)?;

            let pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None)?;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device.allocate_command_buffers(&command_buffer_allocate_info)?;

            let [command_buffer] = command_buffers
                .try_into()
                .map_err(|_| anyhow::anyhow!("Expected 1 command buffer"))?;

            let present_images = swapchain_device.get_swapchain_images(swapchain)?;

            let fence_create_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

            let command_buffer_fence = device.create_fence(&fence_create_info, None)?;

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore =
                device.create_semaphore(&semaphore_create_info, None)?;

            let rendering_complete_semaphore =
                device.create_semaphore(&semaphore_create_info, None)?;

            Ok(Self {
                entry,
                instance,
                device,
                queue_family_index,
                physical_device,
                surface_loader: surface_instance,
                surface,
                swapchain_device,
                swapchain,
                present_queue,
                present_images,
                present_complete_semaphore,
                rendering_complete_semaphore,
                command_buffer,
                command_buffer_fence,
                pool,
            })
        }
    }

    fn render(&self) -> Result<()> {
        unsafe {
            self.device
                .wait_for_fences(&[self.command_buffer_fence], true, u64::MAX)?;

            self.device.reset_fences(&[self.command_buffer_fence])?;

            self.device.reset_command_buffer(
                self.command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )?;

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(self.command_buffer, &command_buffer_begin_info)?;

            let (image_index, _) = self.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.present_complete_semaphore,
                vk::Fence::null(),
            )?;

            let present_image = self.present_images[image_index as usize];

            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let barrier_to_color = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(present_image)
                .subresource_range(subresource_range)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);

            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_color],
            );

            let clear_color_value = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            };

            self.device.cmd_clear_color_image(
                self.command_buffer,
                present_image,
                vk::ImageLayout::GENERAL,
                &clear_color_value,
                &[subresource_range],
            );

            // Transition image to PRESENT_SRC_KHR for presentation
            let barrier_to_present = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(self.present_images[image_index as usize])
                .subresource_range(subresource_range)
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::empty());

            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_present],
            );

            self.device.end_command_buffer(self.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(std::slice::from_ref(&self.present_complete_semaphore))
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])
                .command_buffers(std::slice::from_ref(&self.command_buffer))
                .signal_semaphores(std::slice::from_ref(&self.rendering_complete_semaphore));

            self.device.queue_submit(
                self.present_queue,
                &[submit_info],
                self.command_buffer_fence,
            )?;

            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(std::slice::from_ref(&self.rendering_complete_semaphore))
                .swapchains(std::slice::from_ref(&self.swapchain))
                .image_indices(std::slice::from_ref(&image_index));

            self.swapchain_device
                .queue_present(self.present_queue, &present_info)?;

            self.device
                .wait_for_fences(&[self.command_buffer_fence], true, u64::MAX)?;

            Ok(())
        }
    }
}
