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

            let memory_properties = parameters
                .vk_instance
                .get_physical_device_memory_properties(parameters.physical_device);

            const INDEX_NONE: i32 = -1;
            let mut memory_type_index = INDEX_NONE;
            for i in 0..memory_properties.memory_type_count {
                let is_allowed = memory_requirements.memory_type_bits & (1 << i) != 0;
                let is_desired = memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(parameters.buffer_properties);
                if is_allowed && is_desired {
                    memory_type_index = i as i32;
                    break;
                }
            }
            if memory_type_index == INDEX_NONE {
                return Err(anyhow::anyhow!("Failed to find suitable memory type"));
            }

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
        unsafe {
            let transfer_command_buffer = *device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_pool(transfer_command_pool)
                        .command_buffer_count(1),
                )?
                .first() // requested one buffer, so we're sure there's a first element
                .unwrap();

            device.begin_command_buffer(
                transfer_command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            device.cmd_copy_buffer(
                transfer_command_buffer,
                src_buffer,
                dst_buffer,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(buffer_size)],
            );

            device.end_command_buffer(transfer_command_buffer)?;

            device.queue_submit(
                transfer_queue,
                &[vk::SubmitInfo::default().command_buffers(&[transfer_command_buffer])],
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(transfer_queue)?;

            device.free_command_buffers(transfer_command_pool, &[transfer_command_buffer]);
        }

        Ok(())
    }
}
