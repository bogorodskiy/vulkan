// TODO: better module organization to avoid super
use super::vulkan_utilities::{CreateBufferParameters, VulkanUtilities};
use crate::app::engine::renderer::vertex;
use ash::vk;

pub struct Mesh {
    vertex_count: u32,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_count: u32,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
}

impl Mesh {
    pub fn default() -> Self {
        Self {
            vertex_count: 0,
            vertex_buffer: vk::Buffer::default(),
            vertex_buffer_memory: vk::DeviceMemory::default(),
            index_count: 0,
            index_buffer: vk::Buffer::default(),
            index_buffer_memory: vk::DeviceMemory::default(),
        }
    }

    pub fn new(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        vk_instance: &ash::Instance,
        transfer_queue: vk::Queue,
        transfer_command_pool: vk::CommandPool,
        vertices: &[vertex::Vertex],
        indices: &[u32],
    ) -> anyhow::Result<Self> {
        let mut result = Self::default();

        result.vertex_count = vertices.len() as u32;
        result.create_vertex_buffer(
            device,
            physical_device,
            vk_instance,
            transfer_queue,
            transfer_command_pool,
            vertices,
        )?;

        result.index_count = indices.len() as u32;
        result.create_index_buffer(
            device,
            physical_device,
            vk_instance,
            transfer_queue,
            transfer_command_pool,
            indices,
        )?;

        Ok(result)
    }

    pub fn get_vertex_buffer(&self) -> vk::Buffer {
        self.vertex_buffer
    }

    pub fn get_vertex_count(&self) -> u32 {
        self.vertex_count
    }

    fn create_vertex_buffer(
        &mut self,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        vk_instance: &ash::Instance,
        transfer_queue: vk::Queue,
        transfer_command_pool: vk::CommandPool,
        vertices: &[vertex::Vertex],
    ) -> anyhow::Result<bool> {
        let buffer_size = size_of_val(vertices) as vk::DeviceSize;

        // Temporary buffer to "stage" vertex data before transferring to GPU
        let staging_buffer_parameters = CreateBufferParameters {
            device: device,
            physical_device: physical_device,
            vk_instance: vk_instance,
            buffer_size: buffer_size,
            buffer_usage: vk::BufferUsageFlags::TRANSFER_SRC,
            buffer_properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        };

        unsafe {
            let staging_buffer_result = VulkanUtilities::create_buffer(staging_buffer_parameters)?;

            let data = device.map_memory(
                staging_buffer_result.buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;

            // Copy vertex data to mapped memory
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                data as *mut u8,
                buffer_size as usize,
            );

            device.unmap_memory(staging_buffer_result.buffer_memory);

            // Create a buffer with TRANSFER_DST to mark as the recipient of the transfer data (also VERTEX_BUFFER)
            // Buffer memory is to be DEVICE_LOCAL meaning the memory is on the GPU and only accessible but it and not the CPU (host)
            let vertex_buffer_parameters = CreateBufferParameters {
                device,
                physical_device,
                vk_instance,
                buffer_size: buffer_size,
                buffer_usage: vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::VERTEX_BUFFER,
                buffer_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            };

            let vertex_buffer_result = VulkanUtilities::create_buffer(vertex_buffer_parameters)?;

            self.vertex_buffer = vertex_buffer_result.buffer;
            self.vertex_buffer_memory = vertex_buffer_result.buffer_memory;

            // Copy staging buffer to vertex buffer on GPU
            VulkanUtilities::copy_buffer(
                device,
                transfer_queue,
                transfer_command_pool,
                staging_buffer_result.buffer,
                self.vertex_buffer,
                buffer_size,
            )?;

            device.destroy_buffer(staging_buffer_result.buffer, None);
            device.free_memory(staging_buffer_result.buffer_memory, None);
        }

        Ok(true)
    }

    pub fn get_index_buffer(&self) -> vk::Buffer {
        self.index_buffer
    }

    pub fn get_index_count(&self) -> u32 {
        self.index_count
    }

    fn create_index_buffer(
        &mut self,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        vk_instance: &ash::Instance,
        transfer_queue: vk::Queue,
        transfer_command_pool: vk::CommandPool,
        indices: &[u32],
    ) -> anyhow::Result<bool> {
        // TODO: check if the size is correct
        let buffer_size = size_of_val(indices) as vk::DeviceSize;

        let staging_buffer_parameters = CreateBufferParameters {
            device,
            physical_device,
            vk_instance,
            buffer_size: buffer_size,
            buffer_usage: vk::BufferUsageFlags::TRANSFER_SRC,
            buffer_properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        };

        unsafe {
            let staging_buffer_result = VulkanUtilities::create_buffer(staging_buffer_parameters)?;

            let data = device.map_memory(
                staging_buffer_result.buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;

            // Copy index data to mapped memory
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                data as *mut u8,
                buffer_size as usize,
            );

            device.unmap_memory(staging_buffer_result.buffer_memory);

            // Create a buffer for index data on the GPU access-only area
            let index_buffer_parameters = CreateBufferParameters {
                device,
                physical_device,
                vk_instance,
                buffer_size: buffer_size,
                buffer_usage: vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDEX_BUFFER,
                buffer_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            };

            let index_buffer_result = VulkanUtilities::create_buffer(index_buffer_parameters)?;
            self.index_buffer = index_buffer_result.buffer;
            self.index_buffer_memory = index_buffer_result.buffer_memory;

            // Copy staging buffer to index buffer on GPU
            VulkanUtilities::copy_buffer(
                device,
                transfer_queue,
                transfer_command_pool,
                staging_buffer_result.buffer,
                self.index_buffer,
                buffer_size,
            )?;

            device.destroy_buffer(staging_buffer_result.buffer, None);
            device.free_memory(staging_buffer_result.buffer_memory, None);
        }

        Ok(true)
    }

    pub fn destroy_buffers(&mut self, device: &ash::Device) {
        assert_ne!(
            self.vertex_count, 0,
            "destroy_buffers was called on an empty mesh"
        );
        assert_ne!(
            self.index_count, 0,
            "destroy_buffers was called when indices are not present"
        );

        unsafe {
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_buffer_memory, None);

            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.index_buffer_memory, None);
        }
        self.vertex_count = 0;
        self.index_count = 0;
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        assert_eq!(
            self.vertex_count, 0,
            "vertex_count should be zero at this point. Make sure 'destroy_vertex_buffer' was successfully called before the dtor");
    }
}
