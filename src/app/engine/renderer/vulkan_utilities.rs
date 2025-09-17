use ash::vk;
use nalgebra as na;

#[derive(Copy, Clone)]
pub struct CreateBufferParameters<'a> {
    pub device: &'a ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub vk_instance: &'a ash::Instance,
    pub buffer_size: vk::DeviceSize,
    pub buffer_usage: vk::BufferUsageFlags,
    pub buffer_properties: vk::MemoryPropertyFlags,
}

pub struct CreateBufferResult {
    pub buffer: vk::Buffer,
    pub buffer_memory: vk::DeviceMemory,
}

pub struct UBOViewProjection {
    pub projection: na::Matrix4<f32>,
    pub view: na::Matrix4<f32>,
}

// Uniform buffer object model
#[derive(Copy, Clone)]
pub struct Model {
    model: na::Matrix4<f32>,
}

impl Model {
    pub fn default() -> Self {
        Self {
            model: na::Matrix4::identity(),
        }
    }

    pub fn new(model: na::Matrix4<f32>) -> Self {
        Self { model }
    }
}

impl Default for UBOViewProjection {
    fn default() -> Self {
        Self {
            projection: na::Matrix4::identity(),
            view: na::Matrix4::identity(),
        }
    }
}

pub struct VulkanUtilities;

impl VulkanUtilities {
    pub fn create_buffer(parameters: CreateBufferParameters) -> anyhow::Result<CreateBufferResult> {
        unsafe {
            let buffer = parameters.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(parameters.buffer_size)
                    .usage(parameters.buffer_usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let memory_requirements = parameters.device.get_buffer_memory_requirements(buffer);

            let memory_type_index = Self::find_memory_type_index(
                &parameters.vk_instance,
                parameters.physical_device,
                memory_requirements.memory_type_bits,
                parameters.buffer_properties,
            )?;

            let memory_alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(memory_requirements.size)
                .memory_type_index(memory_type_index as u32);

            let buffer_memory = parameters
                .device
                .allocate_memory(&memory_alloc_info, None)?;
            parameters
                .device
                .bind_buffer_memory(buffer, buffer_memory, 0)?;

            Ok(CreateBufferResult {
                buffer,
                buffer_memory,
            })
        }
    }

    pub fn copy_buffer(
        device: &ash::Device,
        transfer_queue: vk::Queue,
        transfer_command_pool: vk::CommandPool,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        buffer_size: vk::DeviceSize,
    ) -> anyhow::Result<()> {
        // Command buffer to hold the transfer commands
        let transfer_command_buffer = Self::begin_command_buffer(device, transfer_command_pool)?;

        unsafe {
            device.cmd_copy_buffer(
                transfer_command_buffer,
                src_buffer,
                dst_buffer,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(buffer_size)],
            );
        }

        Self::end_and_submit_command_buffer(
            device,
            transfer_command_pool,
            transfer_queue,
            transfer_command_buffer,
        )?;

        Ok(())
    }

    pub fn copy_image_buffer(
        device: &ash::Device,
        transfer_queue: vk::Queue,
        transfer_command_pool: vk::CommandPool,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        width: u32,
        height: u32,
    ) -> anyhow::Result<()> {
        let transfer_command_buffer = Self::begin_command_buffer(device, transfer_command_pool)?;

        unsafe {
            device.cmd_copy_buffer_to_image(
                transfer_command_buffer,
                src_buffer,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0) // for data spacing
                    .buffer_image_height(0) // for data spacing
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D::default()) // 0, 0, 0
                    .image_extent(vk::Extent3D::default().width(width).height(height).depth(1))],
            );
        }

        Self::end_and_submit_command_buffer(
            device,
            transfer_command_pool,
            transfer_queue,
            transfer_command_buffer,
        )?;

        Ok(())
    }

    pub fn transition_image_layout(
        device: &ash::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) -> anyhow::Result<()> {
        let command_buffer = Self::begin_command_buffer(device, command_pool)?;

        let mut image_memory_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let mut src_stage = vk::PipelineStageFlags::empty();
        let mut dst_stage = vk::PipelineStageFlags::empty();

        // if transitioning from new image to image ready to receive data...
        if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            // memory access stage transition must after...
            image_memory_barrier.src_access_mask = vk::AccessFlags::empty();
            // memory access stage transition must before...
            image_memory_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;

            src_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            dst_stage = vk::PipelineStageFlags::TRANSFER;
        }
        // if transitioning from transfer destination to shader readable...
        else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_memory_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            src_stage = vk::PipelineStageFlags::TRANSFER;
            dst_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        }

        unsafe {
            // first two stage flags match to Src and Dst access mask
            device.cmd_pipeline_barrier(
                command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_memory_barrier],
            );
        }

        Self::end_and_submit_command_buffer(device, command_pool, queue, command_buffer)?;

        Ok(())
    }

    fn begin_command_buffer(
        device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> anyhow::Result<vk::CommandBuffer> {
        // Command buffer to hold the transfer commands
        unsafe {
            let command_buffer = *device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_pool(command_pool)
                        .command_buffer_count(1),
                )?
                .first() // requested one buffer, so we're sure there's a first element
                .unwrap();

            device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            Ok(command_buffer)
        }
    }

    fn end_and_submit_command_buffer(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        command_buffer: vk::CommandBuffer,
    ) -> anyhow::Result<()> {
        unsafe {
            device.end_command_buffer(command_buffer)?;

            device.queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(queue)?;

            device.free_command_buffers(command_pool, &[command_buffer]);
        }
        Ok(())
    }

    pub fn find_memory_type_index(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        type_bits: u32,
        prop_flags: vk::MemoryPropertyFlags,
    ) -> anyhow::Result<u32> {
        const INDEX_NONE: i32 = -1;
        let mut memory_type_index = INDEX_NONE;

        unsafe {
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);

            for i in 0..memory_properties.memory_type_count {
                let is_allowed = type_bits & (1 << i) != 0;
                let is_desired = memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(prop_flags);
                if is_allowed && is_desired {
                    memory_type_index = i as i32;
                    break;
                }
            }
        }

        if memory_type_index == INDEX_NONE {
            return Err(anyhow::anyhow!("Failed to find suitable memory type"));
        }

        Ok(memory_type_index as u32)
    }
}
