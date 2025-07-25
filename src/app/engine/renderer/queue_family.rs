pub struct QueueFamilyIndices {
    // If we have a graphics family, we have a transfer family by default
    pub graphics_family: i32,
    pub presentation_family: i32,
}

impl QueueFamilyIndices {
    pub fn default() -> Self {
        Self {
            graphics_family: -1,
            presentation_family: -1,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.graphics_family >= 0 && self.presentation_family >= 0
    }
}
