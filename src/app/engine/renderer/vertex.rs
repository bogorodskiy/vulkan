use nalgebra as na;

pub struct Vertex {
    pub position: na::Vector3<f32>,
    pub color: na::Vector3<f32>,
}

impl Vertex {
    pub fn new(in_position: na::Vector3<f32>, in_color: na::Vector3<f32>) -> Self {
        Vertex {
            position: in_position,
            color: in_color,
        }
    }
}
