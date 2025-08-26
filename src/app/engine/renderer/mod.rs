mod mesh;
mod queue_family;
mod swapchain;
mod vertex;
mod vulkan_device;
mod vulkan_utilities;

use ash::vk;
use ash::vk::MAX_EXTENSION_NAME_SIZE;
use nalgebra as na;
use std::collections::HashSet;
use std::ffi::CStr;
use std::io;
use std::os::raw::c_char;
use std::sync::Arc;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::app::engine::renderer::vulkan_utilities::{
    CreateBufferParameters, Model, UBOViewProjection, VulkanUtilities,
};
use queue_family::QueueFamilyIndices;
use swapchain::{SwapchainDetails, SwapchainImage};
use vulkan_device::VulkanDevice;

const SHADERS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/res/shaders/");
const MAX_FRAME_DRAWS: usize = 2;

pub struct VulkanRenderer {
    // Vulkan components
    instance: ash::Instance,
    window: Arc<Window>,
    entry: ash::Entry, // entry is required, otherwise it will be dropped before we destroy the instance
    surface_extension: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    main_device: VulkanDevice,
    current_frame: usize,

    // Queues
    graphics_queue: vk::Queue,
    presentation_queue: vk::Queue,
    queue_family_indices: QueueFamilyIndices,

    // Swapchain
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<SwapchainImage>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_buffers: Vec<vk::CommandBuffer>,

    // Swapchain utilities
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,

    // Descriptors
    descriptor_set_layout: vk::DescriptorSetLayout,
    push_constant_range: vk::PushConstantRange,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // One buffer per every command buffer
    view_projection_uniform_buffers: Vec<vk::Buffer>,
    view_projection_uniform_buffers_memory: Vec<vk::DeviceMemory>,

    // Synchronization
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    draw_fences: Vec<vk::Fence>,

    // Pipeline
    graphics_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,

    // Pools
    graphics_command_pool: vk::CommandPool,

    // Scene objects
    mesh_list: Vec<mesh::Mesh>,

    // Scene settings
    ubo_view_projection: UBOViewProjection,
}

impl VulkanRenderer {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        unsafe {
            let entry = ash::Entry::load()?;

            let raw_display_handle = window.display_handle()?.as_raw();
            let required_instance_extensions =
                ash_window::enumerate_required_extensions(raw_display_handle)?;

            let instance_properties = entry.enumerate_instance_extension_properties(None)?;
            if !Self::check_extension_support(instance_properties, required_instance_extensions)? {
                Err(anyhow::anyhow!(
                    "One of the required instance extensions is not supported"
                ))?
            }

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(
                        &vk::ApplicationInfo::default()
                            .application_name(c"Vulkan Demo")
                            .application_version(vk::make_api_version(1, 0, 0, 0))
                            .engine_name(c"Vulkan Demo Engine")
                            .engine_version(vk::make_api_version(1, 0, 0, 0))
                            .api_version(vk::API_VERSION_1_3),
                    )
                    .enabled_extension_names(required_instance_extensions),
                None,
            )?;

            let surface_extension = ash::khr::surface::Instance::new(&entry, &instance);
            let mut result = VulkanRenderer {
                instance: instance,
                window: window,
                entry: entry,
                main_device: VulkanDevice {
                    required_device_extensions: [vk::KHR_SWAPCHAIN_NAME.as_ptr()],
                    physical_device: Default::default(),
                    logical_device: None,
                    swapchain_device: None,
                },
                current_frame: 0,
                mesh_list: Default::default(),
                graphics_queue: Default::default(),
                presentation_queue: Default::default(),
                surface_extension: surface_extension,
                surface: Default::default(),
                queue_family_indices: QueueFamilyIndices::default(),
                swapchain: Default::default(),
                swapchain_images: Default::default(),
                swapchain_framebuffers: Default::default(),
                command_buffers: Default::default(),
                image_available_semaphores: Default::default(),
                render_finished_semaphores: Default::default(),
                draw_fences: Default::default(),
                swapchain_image_format: vk::Format::UNDEFINED,
                swapchain_extent: Default::default(),
                descriptor_set_layout: Default::default(),
                push_constant_range: Default::default(),
                descriptor_pool: Default::default(),
                descriptor_sets: Default::default(),
                view_projection_uniform_buffers: Default::default(),
                view_projection_uniform_buffers_memory: Default::default(),
                graphics_pipeline: Default::default(),
                pipeline_layout: Default::default(),
                render_pass: Default::default(),
                graphics_command_pool: Default::default(),
                ubo_view_projection: Default::default(),
            };

            result.create_and_set_surface()?;
            result.find_and_set_physical_device()?;
            // We can cache queue_family_indices when the physical device is ready
            result.queue_family_indices =
                result.get_queue_family_indices(result.main_device.physical_device)?;
            result.create_and_set_logical_device()?;
            result.create_and_set_swapchain_device()?;
            result.create_and_set_swapchain()?;
            result.create_render_pass()?;
            result.create_descriptor_set_layout()?;
            result.create_push_constant_range()?;
            result.create_graphics_pipeline()?;
            result.create_framebuffers()?;
            result.create_command_pool()?;
            result.create_view_projection()?;
            result.create_scene_objects()?;
            result.create_command_buffers()?;
            result.create_uniform_buffers()?;
            result.create_descriptor_pool()?;
            result.create_descriptor_sets()?;
            result.create_synchronization_objects()?;

            Ok(result)
        }
    }

    pub fn update_model(&mut self, model_index: usize, new_model: na::Matrix4<f32>) {
        if self.mesh_list.len() <= model_index {
            return;
        }

        self.mesh_list[model_index].set_model(Model::new(new_model));
    }

    pub fn draw(&mut self) {
        // Get index of the next image to be drawn to, and signal semaphore when ready to be drawn to
        // Common scenarios that can cause is_suboptimal to be true:
        //   - Window resize
        //   - Display mode changes
        //   - Window movement between displays with different properties

        unsafe {
            let logical_device = self.main_device.get_logical_device();

            // Wait for the given fence to signal (open) from the last draw before continuing
            logical_device
                .wait_for_fences(&[self.draw_fences[self.current_frame]], true, u64::MAX)
                .unwrap();
            // Manually reset (close) fences
            logical_device
                .reset_fences(&[self.draw_fences[self.current_frame]])
                .unwrap();

            // TODO: temp fix for this validation error:
            // vkQueueSubmit(): pSubmits[0].pSignalSemaphores[0] (VkSemaphore 0x530000000053) is being signaled by VkQueue 0x1a3d41f9d80,
            // but it may still be in use by VkSwapchainKHR 0x80000000008.
            // Here are the most recently acquired image indices: [0], 1, 2.
            logical_device
                .queue_wait_idle(self.presentation_queue)
                .unwrap();

            let (image_index, _is_suboptimal) = self
                .main_device
                .get_swapchain_device()
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available_semaphores[self.current_frame],
                    vk::Fence::null(),
                )
                .unwrap();
            // TODO: handle is_suboptimal

            self.record_commands(image_index as usize).unwrap();
            self.update_uniform_buffers(image_index).unwrap();

            // Submit the command buffer to the queue
            self.main_device
                .get_logical_device()
                .queue_submit(
                    self.graphics_queue,
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(&[self.image_available_semaphores[self.current_frame]])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(&[self.command_buffers[image_index as usize]])
                        .signal_semaphores(&[self.render_finished_semaphores[self.current_frame]])],
                    self.draw_fences[self.current_frame],
                )
                .unwrap();

            // Present the rendered image to the screen
            let _is_suboptimal = self
                .main_device
                .get_swapchain_device()
                .queue_present(
                    self.presentation_queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[self.render_finished_semaphores[self.current_frame]])
                        .swapchains(&[self.swapchain])
                        .image_indices(&[image_index]),
                )
                .unwrap();
            // TODO: handle is_suboptimal
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAME_DRAWS;
    }

    pub fn resize(&mut self) -> anyhow::Result<()> {
        // TODO: resize swapchain
        Ok(())
    }

    fn check_extension_support(
        properties: Vec<vk::ExtensionProperties>,
        check_extensions: &[*const c_char],
    ) -> anyhow::Result<bool> {
        unsafe {
            for &check_extension in check_extensions.iter() {
                let check_extension_c_str = CStr::from_ptr(check_extension);
                let check_extension_bytes = check_extension_c_str.to_bytes();

                let mut has_extension = false;
                for property in properties.iter() {
                    // extra byte logic to compare check_extension and property.extension_name
                    // without extra heap allocations
                    const NULL_TERMINATOR: c_char = 0;
                    let nul_pos = property
                        .extension_name
                        .iter()
                        .position(|&c| c == NULL_TERMINATOR)
                        .unwrap_or(MAX_EXTENSION_NAME_SIZE);
                    let chars_from_name = &property.extension_name[..nul_pos];
                    let bytes_from_name: &[u8] = std::slice::from_raw_parts(
                        chars_from_name.as_ptr() as *const u8,
                        chars_from_name.len(),
                    );

                    if bytes_from_name == check_extension_bytes {
                        has_extension = true;
                        break;
                    }
                }

                if !has_extension {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn create_and_set_surface(&mut self) -> anyhow::Result<()> {
        let raw_display_handle = self.window.display_handle()?.as_raw();
        let raw_window_handle = self.window.window_handle()?.as_raw();

        unsafe {
            self.surface = ash_window::create_surface(
                &self.entry,
                &self.instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )?;
            Ok(())
        }
    }

    fn find_and_set_physical_device(&mut self) -> anyhow::Result<()> {
        unsafe {
            for &physical_device in self.instance.enumerate_physical_devices()?.iter() {
                if self.check_physical_device_suitable(physical_device)? {
                    self.main_device.physical_device = physical_device;

                    return Ok(());
                }
            }
        }
        Err(anyhow::anyhow!("Unable to fund a suitable physical device"))?
    }

    fn check_physical_device_suitable(
        &mut self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<bool> {
        unsafe {
            let extension_properties = self
                .instance
                .enumerate_device_extension_properties(physical_device)?;
            if !Self::check_extension_support(
                extension_properties,
                &self.main_device.required_device_extensions,
            )? {
                return Ok(false);
            }

            // let device_features = self.instance.get_physical_device_features(device);
            let queue_family_indices = self.get_queue_family_indices(physical_device)?;
            let device_properties = self
                .instance
                .get_physical_device_properties(physical_device);
            let swapchain_details = self.get_swapchain_details(physical_device)?;
            let suitable = queue_family_indices.is_valid()
                && swapchain_details.is_valid()
                && device_properties.api_version >= vk::API_VERSION_1_3;
            Ok(suitable)
        }
    }

    fn get_queue_family_indices(
        &mut self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<QueueFamilyIndices> {
        let mut result = QueueFamilyIndices {
            graphics_family: -1,
            presentation_family: -1,
        };

        unsafe {
            let queue_family_properties = self
                .instance
                .get_physical_device_queue_family_properties(physical_device);

            for (index, queue_family) in queue_family_properties.iter().enumerate() {
                let supports_graphics = queue_family.queue_count > 0
                    && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                if supports_graphics {
                    result.graphics_family = index as i32;
                }

                let supports_presentation =
                    self.surface_extension.get_physical_device_surface_support(
                        physical_device,
                        index as u32,
                        self.surface,
                    )?;
                if supports_presentation {
                    result.presentation_family = index as i32;
                }

                if result.is_valid() {
                    return Ok(result);
                }
            }
        }

        Err(anyhow::anyhow!("Unable to find a suitable queue family"))
    }

    fn get_swapchain_details(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<SwapchainDetails> {
        unsafe {
            let surface_capabilities = self
                .surface_extension
                .get_physical_device_surface_capabilities(physical_device, self.surface)?;

            let formats = self
                .surface_extension
                .get_physical_device_surface_formats(physical_device, self.surface)?;

            let presentation_modes = self
                .surface_extension
                .get_physical_device_surface_present_modes(physical_device, self.surface)?;

            Ok(SwapchainDetails {
                surface_capabilities,
                formats,
                presentation_modes,
            })
        }
    }

    fn create_and_set_logical_device(&mut self) -> anyhow::Result<()> {
        unsafe {
            const HIGHEST_QUEUE_PRIORITY: f32 = 1.0;

            let queue_family_indices_set = HashSet::from([
                self.queue_family_indices.graphics_family,
                self.queue_family_indices.presentation_family,
            ]);

            let mut queue_create_infos = Vec::new();
            for queue_family_index in queue_family_indices_set {
                let queue_create_info = vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index as u32)
                    .queue_priorities(&[HIGHEST_QUEUE_PRIORITY]);
                queue_create_infos.push(queue_create_info);
            }

            // Explicitly set the features we want to use to avoid vulkan validation warnings
            let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
                .timeline_semaphore(true)
                .vulkan_memory_model(true)
                .vulkan_memory_model_device_scope(true)
                .buffer_device_address(true)
                .storage_buffer8_bit_access(true);

            let mut device_features2 = vk::PhysicalDeviceFeatures2::default().features(
                vk::PhysicalDeviceFeatures::default()
                    .fragment_stores_and_atomics(true)
                    .vertex_pipeline_stores_and_atomics(true)
                    .shader_int64(true),
            );
            device_features2 = device_features2.push_next(&mut vulkan_12_features);

            let device = self.instance.create_device(
                self.main_device.physical_device,
                &vk::DeviceCreateInfo::default()
                    // sets queue_create_info_count internally
                    .queue_create_infos(&queue_create_infos)
                    // sets enabled_extension_count internally
                    .enabled_extension_names(&self.main_device.required_device_extensions)
                    .push_next(&mut device_features2),
                None,
            )?;

            self.graphics_queue =
                device.get_device_queue(self.queue_family_indices.graphics_family as u32, 0);
            self.presentation_queue =
                device.get_device_queue(self.queue_family_indices.presentation_family as u32, 0);

            self.main_device.logical_device = Some(device);

            Ok(())
        }
    }

    fn create_view_projection(&mut self) -> anyhow::Result<()> {
        self.ubo_view_projection.projection = na::Matrix4::new_perspective(
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
            45.0_f32.to_radians(),
            0.1,
            100.0,
        );
        self.ubo_view_projection.view = na::Matrix4::look_at_rh(
            &na::Point3::new(0.0, 0.0, 2.0), // look from
            &na::Point3::new(0.0, 0.0, 0.0), // look at
            &na::Vector3::new(0.0, 1.0, 0.0),
        );

        // Invert the y-axis in the projection because Vulkan uses a different coordinate system
        // nalgebra works in a similar way to GLM, which was made for OpenGL
        self.ubo_view_projection.projection[(1, 1)] *= -1.0;

        Ok(())
    }

    fn create_scene_objects(&mut self) -> anyhow::Result<()> {
        let left_rect_point_1 = na::Vector3::new(-0.4, 0.4, 0.0);
        let left_rect_point_2 = na::Vector3::new(-0.4, -0.4, 0.0);
        let left_rect_point_3 = na::Vector3::new(0.4, -0.4, 0.0);
        let left_rect_point_4 = na::Vector3::new(0.4, 0.4, 0.0);

        let right_rect_point_1 = na::Vector3::new(-0.25, 0.6, 0.0);
        let right_rect_point_2 = na::Vector3::new(-0.25, -0.6, 0.0);
        let right_rect_point_3 = na::Vector3::new(0.25, -0.6, 0.0);
        let right_rect_point_4 = na::Vector3::new(0.25, 0.6, 0.0);

        let color_1 = na::Vector3::new(1.0, 0.0, 0.0);
        let color_2 = na::Vector3::new(0.0, 1.0, 0.0);
        let color_3 = na::Vector3::new(0.0, 0.0, 1.0);
        let color_4 = na::Vector3::new(1.0, 1.0, 0.0);

        let left_rect_vertices: Vec<vertex::Vertex> = vec![
            vertex::Vertex::new(left_rect_point_1, color_1),
            vertex::Vertex::new(left_rect_point_2, color_2),
            vertex::Vertex::new(left_rect_point_3, color_3),
            vertex::Vertex::new(left_rect_point_4, color_4),
        ];

        let right_rect_vertices: Vec<vertex::Vertex> = vec![
            vertex::Vertex::new(right_rect_point_1, color_1),
            vertex::Vertex::new(right_rect_point_2, color_2),
            vertex::Vertex::new(right_rect_point_3, color_3),
            vertex::Vertex::new(right_rect_point_4, color_4),
        ];

        let indices: Vec<u32> = vec![0, 1, 2, 2, 3, 0];

        let left_mesh = mesh::Mesh::new(
            self.main_device.get_logical_device(),
            self.main_device.physical_device,
            &self.instance,
            self.graphics_queue,
            self.graphics_command_pool,
            left_rect_vertices.as_slice(),
            indices.as_slice(),
        )?;

        let right_mesh = mesh::Mesh::new(
            self.main_device.get_logical_device(),
            self.main_device.physical_device,
            &self.instance,
            self.graphics_queue,
            self.graphics_command_pool,
            right_rect_vertices.as_slice(),
            indices.as_slice(),
        )?;

        self.mesh_list.push(left_mesh);
        self.mesh_list.push(right_mesh);

        Ok(())
    }

    fn create_and_set_swapchain_device(&mut self) -> anyhow::Result<()> {
        let swapchain_device =
            ash::khr::swapchain::Device::new(&self.instance, self.main_device.get_logical_device());
        self.main_device.swapchain_device = Some(swapchain_device);
        Ok(())
    }

    fn create_and_set_swapchain(&mut self) -> anyhow::Result<()> {
        let swapchain_details = self.get_swapchain_details(self.main_device.physical_device)?;

        // find optimal surface values
        let surface_format = Self::choose_surface_format(&swapchain_details.formats)?;
        let presentation_mode =
            Self::choose_presentation_mode(&swapchain_details.presentation_modes)?;
        let extent = Self::choose_swap_extent(
            &swapchain_details.surface_capabilities,
            self.window.inner_size().width,
            self.window.inner_size().height,
        )?;

        let mut image_count = swapchain_details.surface_capabilities.min_image_count + 1; // +1 for to allow triple buffering
        let is_image_count_limited = swapchain_details.surface_capabilities.max_image_count > 0;
        if is_image_count_limited {
            image_count = image_count.clamp(
                swapchain_details.surface_capabilities.min_image_count,
                swapchain_details.surface_capabilities.max_image_count,
            );
        }

        let mut image_sharing_mode = vk::SharingMode::EXCLUSIVE;
        let mut queue_family_indices_vec: Vec<u32> = Vec::new();
        if self.queue_family_indices.graphics_family
            != self.queue_family_indices.presentation_family
        {
            image_sharing_mode = vk::SharingMode::CONCURRENT;
            queue_family_indices_vec.push(self.queue_family_indices.graphics_family as u32);
            queue_family_indices_vec.push(self.queue_family_indices.presentation_family as u32);
        }

        let swapchain_create_info = &vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .pre_transform(swapchain_details.surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(presentation_mode)
            // TODO: test when exclusive and empty. In cpp it should be nullptr
            .queue_family_indices(queue_family_indices_vec.as_ref())
            .clipped(true)
            // If the old swapchain been destroyed and this one replaces it, then link the old one to quickly
            // hand over responsibilities (e.g., window resized which is not supported here)
            .old_swapchain(self.swapchain);

        unsafe {
            self.swapchain_image_format = surface_format.format;
            self.swapchain_extent = extent;
            self.swapchain = self
                .main_device
                .get_swapchain_device()
                .create_swapchain(swapchain_create_info, None)?;
            self.swapchain_images = self
                .main_device
                .get_swapchain_device()
                .get_swapchain_images(self.swapchain)?
                .into_iter()
                .map(|image| {
                    let image_view = Self::create_image_view(
                        self.main_device.get_logical_device(),
                        image,
                        self.swapchain_image_format,
                        vk::ImageAspectFlags::COLOR,
                    )
                    .unwrap();
                    SwapchainImage { image, image_view }
                })
                .collect::<Vec<_>>();
        }

        Ok(())
    }

    fn create_render_pass(&mut self) -> anyhow::Result<()> {
        // Framebuffer data will be stored as an image, but images can be given different data layouts
        // to give optimal use for certain operations

        let subpass_dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
                .src_access_mask(vk::AccessFlags::MEMORY_READ)
                .dst_subpass(0)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                )
                .dependency_flags(vk::DependencyFlags::BY_REGION), // TODO: set to null in the course, validate with Vulkan tools
            vk::SubpassDependency::default()
                .src_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                )
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .dst_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .dependency_flags(vk::DependencyFlags::BY_REGION), // TODO: set to null in the course, validate with Vulkan tools
        ];

        unsafe {
            self.render_pass = self.main_device.get_logical_device().create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[vk::AttachmentDescription::default()
                        .format(self.swapchain_image_format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)])
                    .subpasses(&[vk::SubpassDescription::default()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[vk::AttachmentReference::default()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])])
                    .dependencies(&subpass_dependencies),
                None,
            )?;
        }
        Ok(())
    }

    fn create_descriptor_set_layout(&mut self) -> anyhow::Result<()> {
        unsafe {
            let view_projection_layout_binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX);

            self.descriptor_set_layout = self
                .main_device
                .get_logical_device()
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&[view_projection_layout_binding]),
                    None,
                )?;
        }

        Ok(())
    }

    fn create_push_constant_range(&mut self) -> anyhow::Result<()> {
        self.push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<Model>() as u32);
        Ok(())
    }

    fn create_graphics_pipeline(&mut self) -> anyhow::Result<()> {
        let vertex_shader_code = match std::fs::read(SHADERS_DIR.to_owned() + "vert.spv") {
            Ok(bytes) => bytes,
            Err(_) => {
                return Err(anyhow::anyhow!("Failed to read vertex shader")); // or handle the error differently
            }
        };
        let fragment_shader_code = match std::fs::read(SHADERS_DIR.to_owned() + "frag.spv") {
            Ok(bytes) => bytes,
            Err(_) => {
                return Err(anyhow::anyhow!("Failed to read fragment shader:")); // or handle the error differently
            }
        };

        let vertex_shader_module = self.create_shader_module(&vertex_shader_code)?;
        let fragment_shader_module = self.create_shader_module(&fragment_shader_code)?;

        let entry_point = std::ffi::CString::new("main")?; // should match the entry point in shaders
        let vertex_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&entry_point);
        let fragment_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&entry_point);

        // TODO: to resize, use dynamic viewport. Can resize in command buffer with vkCmdSetViewport
        // See VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
        // VkPipelineDynamicStateCreateInfo

        let binding_description = vk::VertexInputBindingDescription::default()
            .binding(0) // can bind multiple streams of data, this defines which one
            .stride(size_of::<vertex::Vertex>() as u32) // size of a single vertex object
            .input_rate(vk::VertexInputRate::VERTEX); // How to move between data after each vertex object
                                                      //     vk::VertexInputRate::VERTEX - move on to the next vertex
                                                      //     vk::VertexInputRate::INSTANCE - move to a vertex for the next instance

        // How the data for an attribute is defined within a vertex
        let attribute_descriptions = [
            // POSITION
            vk::VertexInputAttributeDescription::default()
                .binding(0) // which binding the data is at (should be the same as above)
                .location(0) // location where data will be read from
                .format(vk::Format::R32G32B32_SFLOAT) // format data will take (also helps define the size of data)
                .offset(memoffset::offset_of!(vertex::Vertex, position) as u32),
            // COLOR
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(memoffset::offset_of!(vertex::Vertex, color) as u32),
        ];

        let logical_device = self.main_device.get_logical_device();
        unsafe {
            self.pipeline_layout = logical_device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[self.descriptor_set_layout])
                    .push_constant_ranges(&[self.push_constant_range]),
                None,
            )?;

            self.graphics_pipeline = logical_device
                .create_graphics_pipelines(
                    vk::PipelineCache::default(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&[vertex_stage_create_info, fragment_stage_create_info])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_binding_descriptions(&[binding_description])
                                .vertex_attribute_descriptions(&attribute_descriptions),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                .primitive_restart_enable(false),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewports(&[vk::Viewport::default()
                                    .width(self.swapchain_extent.width as f32)
                                    .height(self.swapchain_extent.height as f32)
                                    .min_depth(0.0)
                                    .max_depth(1.0)])
                                .scissors(&[vk::Rect2D::default().extent(self.swapchain_extent)]),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                // Change if fragments beyond near/far planes are clipped (default) or clamped to plane
                                // it also should be in the physical device features (depthClamp)
                                .depth_clamp_enable(false)
                                // Whether to discard data and skip rasterizer. Never creates fragments, only suitable for
                                // the pipeline without frame buffer output
                                .rasterizer_discard_enable(false)
                                .polygon_mode(vk::PolygonMode::FILL)
                                .cull_mode(vk::CullModeFlags::BACK)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .line_width(1.0)
                                .depth_bias_enable(false),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .sample_shading_enable(false)
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        // blending equation: (scrColorBlendFactor * newColor) * colorBlendOperation * (dstColorBlendFactor * oldColor)
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                                    .blend_enable(true)
                                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                                    .color_blend_op(vk::BlendOp::ADD)
                                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                                    .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                                    .alpha_blend_op(vk::BlendOp::ADD)])
                                .logic_op_enable(false),
                        )
                        // .depth_stencil_state(
                        //     // TODO: implement later
                        //     &vk::PipelineDepthStencilStateCreateInfo::default(),
                        // )
                        // .dynamic_state(
                        //     &vk::PipelineDynamicStateCreateInfo::default()
                        //         .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
                        // )
                        .layout(self.pipeline_layout)
                        .render_pass(self.render_pass)
                        .subpass(0)],
                    // End of GraphicsPipelineCreateInfo
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
        }

        // Destroy shader modules, no longer needed after the pipeline created
        unsafe {
            logical_device.destroy_shader_module(vertex_shader_module, None);
            logical_device.destroy_shader_module(fragment_shader_module, None);
        }

        Ok(())
    }

    fn create_framebuffers(&mut self) -> anyhow::Result<()> {
        for swapchain_image in self.swapchain_images.iter() {
            unsafe {
                let frame_buffer = self.main_device.get_logical_device().create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass) // render pass layout Framebuffer will be used with
                        .attachments(&[swapchain_image.image_view])
                        .width(self.swapchain_extent.width)
                        .height(self.swapchain_extent.height)
                        .layers(1),
                    None,
                )?;
                self.swapchain_framebuffers.push(frame_buffer);
            }
        }

        Ok(())
    }

    fn create_command_pool(&mut self) -> anyhow::Result<()> {
        // Create a Graphics Queue Family Command Pool

        unsafe {
            self.graphics_command_pool =
                self.main_device.get_logical_device().create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.queue_family_indices.graphics_family as u32)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )?;
        }

        Ok(())
    }

    fn create_command_buffers(&mut self) -> anyhow::Result<()> {
        unsafe {
            self.command_buffers = self
                .main_device
                .get_logical_device()
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.graphics_command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY) // Buffer you submit directly to queue, cannot be called by other buffers
                        .command_buffer_count(self.swapchain_framebuffers.len() as u32),
                )?;
        }

        // data is being held by graphics_command_pool, no need to delete anything in the destructor
        Ok(())
    }

    fn create_synchronization_objects(&mut self) -> anyhow::Result<()> {
        self.image_available_semaphores = self.create_semaphore_vec(MAX_FRAME_DRAWS);
        self.render_finished_semaphores = self.create_semaphore_vec(MAX_FRAME_DRAWS);

        self.draw_fences = unsafe {
            std::iter::repeat_with(|| {
                self.main_device
                    .get_logical_device()
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .unwrap()
            })
            .take(MAX_FRAME_DRAWS)
            .collect()
        };

        Ok(())
    }

    fn create_uniform_buffers(&mut self) -> anyhow::Result<()> {
        let view_projection_buffer_size = size_of_val(&self.ubo_view_projection) as vk::DeviceSize;
        let num_uniform_buffers = self.swapchain_images.len();

        self.view_projection_uniform_buffers
            .reserve(num_uniform_buffers);
        self.view_projection_uniform_buffers_memory
            .reserve(num_uniform_buffers);

        // Temporary buffer to "stage" vertex data before transferring to GPU
        let vp_buffer_parameters = CreateBufferParameters {
            device: self.main_device.get_logical_device(),
            physical_device: self.main_device.physical_device,
            vk_instance: &self.instance,
            buffer_size: view_projection_buffer_size,
            buffer_usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            buffer_properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        };

        for _ in 0..num_uniform_buffers {
            let vp_buffer_result = VulkanUtilities::create_buffer(vp_buffer_parameters)?;
            self.view_projection_uniform_buffers
                .push(vp_buffer_result.buffer);
            self.view_projection_uniform_buffers_memory
                .push(vp_buffer_result.buffer_memory);
        }

        Ok(())
    }

    // should be created after uniform_buffers
    fn create_descriptor_pool(&mut self) -> anyhow::Result<()> {
        let pool_sizes = [
            // View projection pools. Sets how many descriptors, not descriptor sets
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(self.view_projection_uniform_buffers.len() as u32),
        ];

        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
            // maximum number of descriptor sets that can be created from this pool
            .max_sets(self.swapchain_images.len() as u32)
            .pool_sizes(&pool_sizes);

        unsafe {
            self.descriptor_pool = self
                .main_device
                .get_logical_device()
                .create_descriptor_pool(&pool_create_info, None)?;
        }

        Ok(())
    }

    // should be created after descriptor_pool
    fn create_descriptor_sets(&mut self) -> anyhow::Result<()> {
        let descriptor_set_count = self.swapchain_images.len();
        let set_layouts = vec![self.descriptor_set_layout; descriptor_set_count];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);

        unsafe {
            self.descriptor_sets = self
                .main_device
                .get_logical_device()
                .allocate_descriptor_sets(&allocate_info)?;
        }

        // update all descriptor sets buffer bindings
        for i in 0..descriptor_set_count {
            //  VIEW PROJECTION DESCRIPTOR ============

            // buffer info and data offset info
            let vp_buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(self.view_projection_uniform_buffers[i])
                .offset(0)
                .range(size_of_val(&self.ubo_view_projection) as vk::DeviceSize);

            // Create a slice containing the buffer info to extend its lifetime
            let vp_buffer_infos = [vp_buffer_info];

            // Data about the connection between the binding and the buffer
            let vp_set_write = vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .buffer_info(&vp_buffer_infos);

            unsafe {
                // Update the descriptor sets with the new buffer/binding info
                self.main_device
                    .get_logical_device()
                    .update_descriptor_sets(&[vp_set_write], &[]);
            }
        }

        Ok(())
    }

    fn update_uniform_buffers(&mut self, image_index: u32) -> anyhow::Result<()> {
        unsafe {
            // Copy VP data
            // Static descriptors are good for data that doesn't change often (e.g., view projection matrix)
            let vp_data = self.main_device.get_logical_device().map_memory(
                self.view_projection_uniform_buffers_memory[image_index as usize],
                0,
                size_of_val(&self.ubo_view_projection) as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(&self.ubo_view_projection),
                vp_data as *mut UBOViewProjection,
                1,
            );
            self.main_device
                .get_logical_device()
                .unmap_memory(self.view_projection_uniform_buffers_memory[image_index as usize]);
        }
        Ok(())
    }

    fn create_semaphore_vec(&self, num_elements: usize) -> Vec<vk::Semaphore> {
        unsafe {
            let result = std::iter::repeat_with(|| {
                self.main_device
                    .get_logical_device()
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap()
            })
            .take(num_elements)
            .collect();

            return result;
        }
    }

    fn create_image_view(
        logical_device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
    ) -> anyhow::Result<vk::ImageView> {
        unsafe {
            // ComponentSwizzle
            let component_mapping: vk::ComponentMapping = vk::ComponentMapping::default()
                // allows remapping of rgba components to other rgba values
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);

            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_flags)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let image_view = logical_device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image) // image to create view for
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(component_mapping)
                    .subresource_range(subresource_range),
                None,
            )?;

            Ok(image_view)
        }
    }

    fn choose_surface_format(
        formats: &[vk::SurfaceFormatKHR],
    ) -> anyhow::Result<vk::SurfaceFormatKHR> {
        let desired_format = vk::Format::R8G8B8A8_UNORM;
        let backup_format = vk::Format::R8G8B8A8_UNORM;
        let desired_color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let all_formats_are_available =
            formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED;

        if all_formats_are_available {
            return Ok(vk::SurfaceFormatKHR {
                format: desired_format,
                color_space: desired_color_space,
            });
        }

        let option_format = formats
            .iter()
            .find(|format| {
                (format.format == desired_format || format.format == backup_format)
                    && format.color_space == desired_color_space
            })
            .copied();

        if let Some(result) = option_format {
            return Ok(result);
        }

        // no desired formats -> return the first one in the list
        if !formats.is_empty() {
            return Ok(formats[0]);
        }

        Err(anyhow::anyhow!("Unable choose a surface format"))
    }

    fn choose_presentation_mode(
        modes: &[vk::PresentModeKHR],
    ) -> anyhow::Result<vk::PresentModeKHR> {
        let desired_mode = vk::PresentModeKHR::MAILBOX;
        let backup_mode = vk::PresentModeKHR::FIFO; // Should be always available
        let option_mode = modes.iter().copied().find(|mode| *mode == desired_mode);
        if let Some(result) = option_mode {
            return Ok(result);
        }

        Ok(backup_mode)
    }

    fn choose_swap_extent(
        surface_capabilities: &vk::SurfaceCapabilitiesKHR,
        window_width: u32,
        window_height: u32,
    ) -> anyhow::Result<vk::Extent2D> {
        if surface_capabilities.current_extent.width != u32::MAX {
            return Ok(surface_capabilities.current_extent);
        }

        let mut result = vk::Extent2D {
            width: window_width,
            height: window_height,
        };

        result.width = result.width.clamp(
            surface_capabilities.min_image_extent.width,
            surface_capabilities.max_image_extent.width,
        );
        result.height = result.height.clamp(
            surface_capabilities.min_image_extent.height,
            surface_capabilities.max_image_extent.height,
        );

        Ok(result)
    }

    fn create_shader_module(&self, code: &[u8]) -> anyhow::Result<vk::ShaderModule> {
        let mut code = io::Cursor::new(code);
        let code = ash::util::read_spv(&mut code)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe {
            self.main_device
                .get_logical_device()
                .create_shader_module(&create_info, None)
        }?;
        Ok(shader_module)
    }

    fn record_commands(&mut self, current_image_index: usize) -> anyhow::Result<()> {
        // information about how to begin a render pass
        let mut clear_values = [vk::ClearValue::default()];
        clear_values[0].color = vk::ClearColorValue {
            float32: [0.6, 0.65, 0.4, 1.0],
        };

        let mut render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .render_area(vk::Rect2D::default().extent(self.swapchain_extent))
            .clear_values(&clear_values);

        let logical_device = self.main_device.get_logical_device();

        // command_buffer is a wrapper around a pointer (handle)
        unsafe {
            render_pass_begin_info.framebuffer = self.swapchain_framebuffers[current_image_index];
            let command_buffer = self.command_buffers[current_image_index];

            // start recording commands to the command buffer
            logical_device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

            logical_device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            // bind pipeline to be used in render pass
            logical_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );

            for (mesh_index, mesh) in self.mesh_list.iter().enumerate() {
                logical_device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    // buffer to bind
                    &[mesh.get_vertex_buffer()],
                    // offsets into buffers being bound
                    &[0 as vk::DeviceSize],
                );

                // only one index buffer, but it can be used for many vertex buffers
                logical_device.cmd_bind_index_buffer(
                    command_buffer,
                    mesh.get_index_buffer(),
                    0,
                    vk::IndexType::UINT32,
                );

                let model_bytes = std::slice::from_raw_parts(
                    self.mesh_list[mesh_index].get_model() as *const Model as *const u8,
                    size_of::<Model>(),
                );

                logical_device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    &model_bytes,
                );

                logical_device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_sets[current_image_index]],
                    &[],
                );

                // Execute pipeline
                logical_device.cmd_draw_indexed(command_buffer, mesh.get_index_count(), 1, 0, 0, 0);
            }

            logical_device.cmd_end_render_pass(command_buffer);

            // stop recording to the command buffer
            logical_device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        // Expected behavior that VulkanRenderer::new succeeded before this function
        let logical_device = self.main_device.get_logical_device();

        unsafe {
            // wait until no actions being run on the device before destroying
            logical_device.device_wait_idle().unwrap();

            logical_device.destroy_descriptor_pool(self.descriptor_pool, None);
            logical_device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            for uniform_buffer in self.view_projection_uniform_buffers.iter() {
                logical_device.destroy_buffer(*uniform_buffer, None);
            }
            for uniform_buffer_memory in self.view_projection_uniform_buffers_memory.iter() {
                logical_device.free_memory(*uniform_buffer_memory, None);
            }

            for mesh in self.mesh_list.iter_mut() {
                mesh.destroy_buffers(logical_device);
            }

            for semaphore in self.render_finished_semaphores.drain(..) {
                logical_device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.image_available_semaphores.drain(..) {
                logical_device.destroy_semaphore(semaphore, None);
            }
            for fence in self.draw_fences.drain(..) {
                logical_device.destroy_fence(fence, None);
            }

            logical_device.destroy_command_pool(self.graphics_command_pool, None);

            // take swapchain_framebuffers from self and leave it empty in self
            let taken_framebuffers = std::mem::take(&mut self.swapchain_framebuffers);
            for frame_buffer in taken_framebuffers {
                logical_device.destroy_framebuffer(frame_buffer, None);
            }

            logical_device.destroy_pipeline(self.graphics_pipeline, None);
            logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
            logical_device.destroy_render_pass(self.render_pass, None);
            for swapchain_image in &self.swapchain_images {
                // swapchain will destroy the image, but we need to destroy image_view because it was created by us
                logical_device.destroy_image_view(swapchain_image.image_view, None);
            }

            if let Some(swapchain_device) = self.main_device.swapchain_device.as_ref() {
                swapchain_device.destroy_swapchain(self.swapchain, None);
            }

            self.surface_extension.destroy_surface(self.surface, None);
            logical_device.destroy_device(None);

            self.instance.destroy_instance(None);
        }
    }
}
