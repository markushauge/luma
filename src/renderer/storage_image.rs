use anyhow::Result;
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use super::Device;

#[derive(Default)]
#[expect(dead_code)]
pub struct StorageImage {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
}

impl Device {
    pub fn create_storage_image(
        &self,
        extent: vk::Extent2D,
        format: vk::Format,
        name: Option<&str>,
    ) -> Result<StorageImage> {
        unsafe {
            let image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let image = self.device.create_image(&image_create_info, None)?;
            let requirements = self.device.get_image_memory_requirements(image);

            let allocation = self.allocate(&AllocationCreateDesc {
                name: name.unwrap_or("Storage Image"),
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedImage(image),
            });

            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())?;

            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let image_view = self
                .device
                .create_image_view(&image_view_create_info, None)?;

            Ok(StorageImage {
                extent,
                format,
                image,
                image_view,
                allocation,
            })
        }
    }

    pub fn destroy_storage_image(&self, image: StorageImage) {
        unsafe {
            self.device.destroy_image_view(image.image_view, None);
            self.device.destroy_image(image.image, None);
            self.free(image.allocation);
        }
    }
}
