use std::{ffi::c_char, sync::Arc, time::Instant};

use anyhow::{Result, anyhow};
use ash::{khr, vk};
use bevy::{
    prelude::*,
    window::{PrimaryWindow, RawHandleWrapper},
};
use bytemuck::{Pod, Zeroable};

use crate::{
    camera::Camera,
    shader::{Shader, ShaderPlugin},
};

#[derive(Resource, Clone)]
pub struct RendererSettings {
    pub resolution_scaling: f32,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            resolution_scaling: 1.0,
        }
    }
}

#[derive(Resource)]
struct ComputeShader(Handle<Shader>);

#[derive(States, Default, Debug, Hash, PartialEq, Eq, Clone)]
enum RendererState {
    #[default]
    Loading,
    Ready,
}

#[derive(Default)]
pub struct RendererPlugin {
    pub settings: RendererSettings,
}

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ShaderPlugin)
            .init_state::<RendererState>()
            .insert_resource(self.settings.clone())
            .add_systems(OnEnter(RendererState::Loading), load_shader)
            .add_systems(
                Update,
                check_shader_loaded.run_if(in_state(RendererState::Loading)),
            )
            .add_systems(OnEnter(RendererState::Ready), setup_renderer)
            .add_systems(Update, render.run_if(in_state(RendererState::Ready)));
    }
}

fn load_shader(mut commands: Commands, asset_server: Res<AssetServer>) {
    let shader = asset_server.load("shaders/main.comp");
    commands.insert_resource(ComputeShader(shader));
}

fn check_shader_loaded(
    compute_shader: Res<ComputeShader>,
    assets: Res<Assets<Shader>>,
    mut next_state: ResMut<NextState<RendererState>>,
) {
    if assets.get(&compute_shader.0).is_some() {
        next_state.set(RendererState::Ready);
    }
}

fn setup_renderer(
    mut commands: Commands,
    windows: Query<(&Window, &RawHandleWrapper), With<PrimaryWindow>>,
    settings: Res<RendererSettings>,
    compute_shader: Res<ComputeShader>,
    assets: Res<Assets<Shader>>,
) -> Result<(), BevyError> {
    let shader = assets.get(&compute_shader.0).unwrap();
    let (window, raw_handles) = windows.single()?;
    let UVec2 { x, y } = window.physical_size();
    let renderer = Renderer::new(raw_handles, x, y, &settings, shader)?;
    commands.insert_resource(renderer);
    Ok(())
}

fn render(
    mut renderer: ResMut<Renderer>,
    query: Query<&Transform, With<Camera>>,
) -> Result<(), BevyError> {
    let camera_transform = query.single()?;
    renderer.render(camera_transform)?;
    Ok(())
}

const MAX_CONCURRENT_FRAMES: u32 = 2;

#[derive(Resource)]
pub struct Renderer {
    device: Device,
    swapchain: Swapchain,
    compute_pipeline: ComputePipeline,
    frames: Vec<Frame>,
    frame_count: usize,
    start_time: Instant,
}

impl Renderer {
    pub fn new(
        raw_handles: &RawHandleWrapper,
        width: u32,
        height: u32,
        settings: &RendererSettings,
        shader: &Shader,
    ) -> Result<Self> {
        let device = Device::new(raw_handles)?;
        let swapchain = Swapchain::new(device.clone(), raw_handles, width, height)?;

        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            width,
            height,
            settings.resolution_scaling,
            shader,
        )?;

        let command_buffers = unsafe {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(MAX_CONCURRENT_FRAMES)
                .command_pool(device.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            device
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)?
        };

        let frames = command_buffers
            .into_iter()
            .map(|command_buffer| Frame::new(&device, command_buffer))
            .collect::<Result<Vec<_>, _>>()?;

        let frame_count = 0;
        let start_time = Instant::now();

        Ok(Self {
            device,
            swapchain,
            compute_pipeline,
            frames,
            frame_count,
            start_time,
        })
    }

    pub fn render(&mut self, camera_transform: &Transform) -> Result<()> {
        let frame_index = self.frame_count % self.frames.len();
        let frame = &self.frames[frame_index];

        self.device.begin_frame(frame)?;

        let (image_index, present_image) = self
            .swapchain
            .acquire_next_image(frame.present_complete_semaphore)?;

        let time_millis = Instant::now().duration_since(self.start_time).as_millis() as u32;

        self.compute_pipeline
            .dispatch(frame, camera_transform, time_millis);

        self.compute_pipeline
            .blit(frame, present_image, self.swapchain.surface_extent);

        self.device.end_frame(frame)?;

        self.swapchain
            .present_image(image_index, frame.rendering_complete_semaphore)?;

        self.frame_count += 1;

        Ok(())
    }
}

#[derive(Clone, Deref)]
struct Device(Arc<DeviceInner>);

#[allow(dead_code)]
struct DeviceInner {
    entry: ash::Entry,
    instance: ash::Instance,
    surface_instance: khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device: ash::Device,
    swapchain_device: khr::swapchain::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
}

impl Device {
    fn new(raw_handles: &RawHandleWrapper) -> Result<Self> {
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

            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let command_pool = device.create_command_pool(&command_pool_create_info, None)?;

            let inner = DeviceInner {
                entry,
                instance,
                surface_instance,
                physical_device,
                queue_family_index,
                device,
                swapchain_device,
                queue,
                command_pool,
            };

            Ok(Self(Arc::new(inner)))
        }
    }

    fn begin_frame(&self, frame: &Frame) -> Result<()> {
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

    fn end_frame(&self, frame: &Frame) -> Result<()> {
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

    fn transition_image(
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

struct Frame {
    command_buffer: vk::CommandBuffer,
    present_complete_semaphore: vk::Semaphore,
    rendering_complete_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl Frame {
    fn new(device: &Device, command_buffer: vk::CommandBuffer) -> Result<Self> {
        unsafe {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .device
                .create_semaphore(&semaphore_create_info, None)?;

            let rendering_complete_semaphore = device
                .device
                .create_semaphore(&semaphore_create_info, None)?;

            let fence_create_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

            let fence = device.device.create_fence(&fence_create_info, None)?;

            Ok(Self {
                command_buffer,
                present_complete_semaphore,
                rendering_complete_semaphore,
                fence,
            })
        }
    }
}

#[allow(dead_code)]
struct Swapchain {
    device: Device,
    surface: vk::SurfaceKHR,
    surface_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    present_images: Vec<vk::Image>,
}

impl Swapchain {
    fn new(
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
                .ok_or_else(|| anyhow::anyhow!("No suitable surface format found"))?;

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
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(std::slice::from_ref(&device.queue_family_index))
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);

            let swapchain = device
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)?;

            let present_images = device.swapchain_device.get_swapchain_images(swapchain)?;

            Ok(Self {
                device,
                surface,
                surface_extent,
                swapchain,
                present_images,
            })
        }
    }

    fn acquire_next_image(
        &self,
        present_complete_semaphore: vk::Semaphore,
    ) -> Result<(u32, vk::Image)> {
        unsafe {
            let (image_index, _) = self.device.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                present_complete_semaphore,
                vk::Fence::null(),
            )?;

            let present_image = self.present_images[image_index as usize];
            Ok((image_index, present_image))
        }
    }

    fn present_image(&mut self, image_index: u32, wait_semaphore: vk::Semaphore) -> Result<()> {
        unsafe {
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(std::slice::from_ref(&wait_semaphore))
                .swapchains(std::slice::from_ref(&self.swapchain))
                .image_indices(std::slice::from_ref(&image_index));

            self.device
                .swapchain_device
                .queue_present(self.device.queue, &present_info)?;

            Ok(())
        }
    }
}

#[allow(dead_code)]
struct ComputePipeline {
    device: Device,
    storage_image: vk::Image,
    storage_image_extent: vk::Extent3D,
    storage_image_memory: vk::DeviceMemory,
    storage_image_view: vk::ImageView,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl ComputePipeline {
    fn new(
        device: Device,
        width: u32,
        height: u32,
        resolution_scaling: f32,
        shader: &Shader,
    ) -> Result<Self> {
        unsafe {
            let width = (width as f32 * resolution_scaling) as u32;
            let height = (height as f32 * resolution_scaling) as u32;

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

            let device_memory_properties = device
                .instance
                .get_physical_device_memory_properties(device.physical_device);

            let image_memory_requirements =
                device.device.get_image_memory_requirements(storage_image);

            let memory_type_index = (0..vk::MAX_MEMORY_TYPES)
                .find(|i| {
                    (image_memory_requirements.memory_type_bits & (1 << i)) != 0
                        && device_memory_properties.memory_types[*i]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .ok_or_else(|| anyhow::anyhow!("No suitable memory type for storage image"))?;

            let memory_allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(image_memory_requirements.size)
                .memory_type_index(memory_type_index as u32);

            let storage_image_memory =
                device.device.allocate_memory(&memory_allocate_info, None)?;

            device
                .device
                .bind_image_memory(storage_image, storage_image_memory, 0)?;

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

            let shader_module_create_info =
                vk::ShaderModuleCreateInfo::default().code(&shader.code);

            let compute_shader_module = device
                .device
                .create_shader_module(&shader_module_create_info, None)?;

            let descriptor_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE);

            let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(std::slice::from_ref(&descriptor_set_layout_binding));

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
                .name(c"main");

            let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
                .stage(shader_stage_create_info)
                .layout(pipeline_layout);

            let pipelines_result = device.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            );

            let pipeline = match pipelines_result {
                Ok(pipelines) => pipelines[0],
                Err((pipelines, err)) => {
                    if !pipelines.is_empty() {
                        pipelines[0]
                    } else {
                        return Err(anyhow::anyhow!(
                            "Failed to create compute pipeline: {:?}",
                            err
                        ));
                    }
                }
            };

            device
                .device
                .destroy_shader_module(compute_shader_module, None);

            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1);

            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(std::slice::from_ref(&pool_size));

            let descriptor_pool = device
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)?;

            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));

            let descriptor_set = device
                .device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)?[0];

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
                storage_image,
                storage_image_extent,
                storage_image_memory,
                storage_image_view,
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_set,
            })
        }
    }

    fn dispatch(&self, frame: &Frame, camera_transform: &Transform, time_millis: u32) {
        unsafe {
            self.device.transition_image(
                frame.command_buffer,
                self.storage_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

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
                &[self.descriptor_set],
                &[],
            );

            let push_constants = PushConstants {
                viewport_width: self.storage_image_extent.width,
                viewport_height: self.storage_image_extent.height,
                camera_translation: camera_transform.translation,
                camera_rotation: Mat3::from_quat(camera_transform.rotation),
                camera_fov: 52.0f32.to_radians(), // TODO: Make configurable
                time_millis,
            };

            self.device.device.cmd_push_constants(
                frame.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            self.device.device.cmd_dispatch(
                frame.command_buffer,
                self.storage_image_extent.width.div_ceil(16),
                self.storage_image_extent.height.div_ceil(16),
                1,
            );
        }
    }

    fn blit(&self, frame: &Frame, present_image: vk::Image, present_image_extent: vk::Extent2D) {
        unsafe {
            self.device.transition_image(
                frame.command_buffer,
                self.storage_image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            self.device.transition_image(
                frame.command_buffer,
                present_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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

            self.device.device.cmd_blit_image(
                frame.command_buffer,
                self.storage_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                present_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[image_blit],
                vk::Filter::LINEAR,
            );

            self.device.transition_image(
                frame.command_buffer,
                self.storage_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );

            self.device.transition_image(
                frame.command_buffer,
                present_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
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
