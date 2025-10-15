use anyhow::{Result, anyhow};
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use super::Device;

#[allow(dead_code)]
pub struct Frame {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub semaphore: vk::Semaphore,
    pub fence: vk::Fence,
    pub storage_image: vk::Image,
    pub storage_image_extent: vk::Extent3D,
    pub storage_image_allocation: Allocation,
    pub storage_image_view: vk::ImageView,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

impl Frame {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        descriptor_set_layout_bindings: &[vk::DescriptorSetLayoutBinding<'_>],
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        unsafe {
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(device.queue_family_index);

            let command_pool = device
                .device
                .create_command_pool(&command_pool_create_info, None)?;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let [command_buffer] = device
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)?
                .try_into()
                .map_err(|_| anyhow!("Failed to allocate exactly one command buffer"))?;

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let semaphore = device
                .device
                .create_semaphore(&semaphore_create_info, None)?;

            let fence_create_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

            let fence = device.device.create_fence(&fence_create_info, None)?;

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
                command_pool,
                command_buffer,
                semaphore,
                fence,
                storage_image,
                storage_image_extent,
                storage_image_allocation,
                storage_image_view,
                descriptor_pool,
                descriptor_set,
            })
        }
    }

    pub fn destroy(self, device: &Device) {
        unsafe {
            device
                .device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .unwrap();

            device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            device
                .device
                .destroy_image_view(self.storage_image_view, None);

            device.device.destroy_image(self.storage_image, None);

            device.free(self.storage_image_allocation).unwrap();

            device.device.destroy_fence(self.fence, None);

            device.device.destroy_semaphore(self.semaphore, None);

            device
                .device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);

            device.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
