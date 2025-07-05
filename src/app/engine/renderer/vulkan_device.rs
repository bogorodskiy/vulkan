use ash::vk;
use std::os::raw::c_char;

pub struct VulkanDevice {
    pub required_device_extensions: [*const c_char; 1],
    pub physical_device: vk::PhysicalDevice,
    pub logical_device: Option<ash::Device>,
    pub swapchain_device: Option<ash::khr::swapchain::Device>,
}

impl VulkanDevice {
    pub fn get_logical_device(&self) -> &ash::Device {
        self.logical_device
            .as_ref()
            .expect("create_and_set_logical_device should be called before calling this method")
    }

    pub fn get_swapchain_device(&self) -> &ash::khr::swapchain::Device {
        self.swapchain_device
            .as_ref()
            .expect("create_and_set_swapchain_device should be called before calling this method")
    }
}
