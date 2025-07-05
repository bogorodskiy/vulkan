use ash::vk;

pub(crate) struct SwapchainDetails {
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub presentation_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainDetails {
    pub fn is_valid(&self) -> bool {
        !self.formats.is_empty() && !self.presentation_modes.is_empty()
    }
}

pub(crate) struct SwapchainImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
}
