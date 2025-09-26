use nalgebra as na;

pub struct Vertex {
    pub position: na::Vector3<f32>,            // (x, y, z)
    pub color: na::Vector3<f32>,               // (x, y, z)
    pub texture_coordinates: na::Vector2<f32>, // (u, v)
}

impl Vertex {
    pub fn new(
        in_position: na::Vector3<f32>,
        in_color: na::Vector3<f32>,
        in_texture_coordinates: na::Vector2<f32>,
    ) -> Self {
        Vertex {
            position: in_position,
            color: in_color,
            texture_coordinates: in_texture_coordinates,
        }
    }
}
