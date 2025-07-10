use crate::app::engine::renderer::vertex;
use crate::app::engine::renderer::vertex::Vertex;
use ash::vk;

pub struct Mesh {
    vertex_count: u32,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
}

impl Mesh {
    pub fn default() -> Self {
        Self {
            vertex_count: 0,
            vertex_buffer: vk::Buffer::default(),
            vertex_buffer_memory: vk::DeviceMemory::default(),
        }
    }

    pub fn new(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        vk_instance: &ash::Instance,
        vertices: &[vertex::Vertex],
    ) -> anyhow::Result<Self> {
        let mut result = Self {
            vertex_count: vertices.len() as u32,
            vertex_buffer: vk::Buffer::default(),
            vertex_buffer_memory: vk::DeviceMemory::default(),
        };

        result.create_vertex_buffer(device, physical_device, vk_instance, vertices)?;
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
        vertices: &[Vertex],
    ) -> anyhow::Result<bool> {
        unsafe {
            let vertex_buffer_size = size_of_val(vertices) as vk::DeviceSize;
            self.vertex_buffer = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(vertex_buffer_size)
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let memory_requirements = device.get_buffer_memory_requirements(self.vertex_buffer);

            let memory_properties =
                vk_instance.get_physical_device_memory_properties(physical_device);

            // Visible to CPU and the memory after being mapped is placed straight into the buffer
            let desired_properties =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            const INDEX_NONE: i32 = -1;
            let mut memory_type_index = INDEX_NONE;
            for i in 0..memory_properties.memory_type_count {
                let is_allowed = memory_requirements.memory_type_bits & (1 << i) != 0;
                let is_desired = memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(desired_properties);
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

            self.vertex_buffer_memory = device.allocate_memory(&memory_alloc_info, None)?;
            device.bind_buffer_memory(self.vertex_buffer, self.vertex_buffer_memory, 0)?;
            let data = device.map_memory(
                self.vertex_buffer_memory,
                0,
                vertex_buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;

            // Copy vertex data to mapped memory
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                data as *mut u8,
                vertex_buffer_size as usize,
            );

            device.unmap_memory(self.vertex_buffer_memory);
        }

        Ok(true)
    }

    pub fn destroy_vertex_buffer(&mut self, device: &ash::Device) {
        assert_ne!(
            self.vertex_count, 0,
            "destroy_vertex_buffer was called on an empty mesh"
        );
        unsafe {
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_buffer_memory, None);
        }
        self.vertex_count = 0;
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        assert_eq!(
            self.vertex_count, 0,
            "vertex_count should be zero at this point. Make sure 'destroy_vertex_buffer' was successfully called before the dtor");
    }
}
