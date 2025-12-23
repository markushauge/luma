use anyhow::{Result, anyhow};
use ash::vk;
use bevy::window::RawHandleWrapper;

use super::Device;

pub struct SwapchainImage {
    pub image: vk::Image,
    pub semaphore: vk::Semaphore,
}

#[allow(dead_code)]
pub struct Swapchain {
    pub device: Device,
    pub surface: vk::SurfaceKHR,
    pub surface_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub present_images: Vec<SwapchainImage>,
    pub present_image_index: u32,
}

impl Swapchain {
    pub fn new(
        device: Device,
        raw_handles: &RawHandleWrapper,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        unsafe {
            let display_handle = raw_handles.get_display_handle();
            let window_handle = raw_handles.get_window_handle();

            let surface = ash_window::create_surface(
                &device.entry,
                &device.instance,
                display_handle,
                window_handle,
                None,
            )?;

            let surface_formats = device
                .surface_instance
                .get_physical_device_surface_formats(device.physical_device, surface)?;

            let surface_format = surface_formats
                .iter()
                .find(|format| {
                    (format.format == vk::Format::B8G8R8A8_UNORM
                        || format.format == vk::Format::R8G8B8A8_UNORM)
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .ok_or_else(|| anyhow!("No suitable surface format found"))?;

            tracing::info!("Using surface format: {:?}", surface_format);

            let surface_capabilities = device
                .surface_instance
                .get_physical_device_surface_capabilities(device.physical_device, surface)?;

            let mut desired_image_count = surface_capabilities.min_image_count + 1;

            if surface_capabilities.max_image_count > 0 {
                desired_image_count = desired_image_count.min(surface_capabilities.max_image_count);
            }

            let surface_extent = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D { width, height },
                _ => surface_capabilities.current_extent,
            };

            let present_mode = device
                .surface_instance
                .get_physical_device_surface_present_modes(device.physical_device, surface)?
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(surface_extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(std::slice::from_ref(&device.queue_family_index))
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);

            let swapchain = device
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)?;

            let present_images = device
                .swapchain_device
                .get_swapchain_images(swapchain)?
                .into_iter()
                .map(|image| {
                    let semaphore_create_info = vk::SemaphoreCreateInfo::default();

                    let semaphore = device
                        .device
                        .create_semaphore(&semaphore_create_info, None)?;

                    Ok(SwapchainImage { image, semaphore })
                })
                .collect::<Result<Vec<_>>>()?;

            let present_image_index = 0;

            Ok(Self {
                device,
                surface,
                surface_extent,
                swapchain,
                present_images,
                present_image_index,
            })
        }
    }

    pub fn acquire_next(&mut self, signal_semaphore: vk::Semaphore) -> Result<()> {
        unsafe {
            let (image_index, _) = self.device.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                signal_semaphore,
                vk::Fence::null(),
            )?;

            self.present_image_index = image_index;
            Ok(())
        }
    }

    pub fn present(&mut self) -> Result<()> {
        let present_image = self.present_image();

        unsafe {
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(std::slice::from_ref(&present_image.semaphore))
                .swapchains(std::slice::from_ref(&self.swapchain))
                .image_indices(std::slice::from_ref(&self.present_image_index));

            self.device
                .swapchain_device
                .queue_present(self.device.queue, &present_info)?;

            Ok(())
        }
    }

    pub fn present_image(&self) -> &SwapchainImage {
        &self.present_images[self.present_image_index as usize]
    }
}
