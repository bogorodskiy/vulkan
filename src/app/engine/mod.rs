mod renderer;

use crate::app::engine::renderer::VulkanRenderer;
use anyhow::Result;
use nalgebra as na;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

pub trait BaseEngine {
    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent);
    fn request_redraw(&mut self);
}

pub struct UninitializedEngine;
impl BaseEngine for UninitializedEngine {
    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowEvent) {
        // do nothing
    }

    fn request_redraw(&mut self) {
        // do nothing
    }
}

pub struct Engine {
    renderer: VulkanRenderer,
    angle: f32,
    last_time: std::time::Instant,
}

impl Engine {
    pub(crate) fn new(event_loop: &ActiveEventLoop) -> Result<Self> {
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan Demo")
            .with_inner_size(LogicalSize::new(800.0, 600.0))
            .with_resizable(false);
        let window = Arc::new(event_loop.create_window(window_attributes)?);
        let renderer = VulkanRenderer::new(window)?;

        Ok(Self {
            renderer: renderer,
            angle: 0.0,
            last_time: std::time::Instant::now(),
        })
    }
}

impl BaseEngine for Engine {
    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.renderer.resize().unwrap();
            }
            WindowEvent::RedrawRequested => {
                self.request_redraw();
            }
            _ => {}
        }

        // WindowEvent::ScaleFactorChanged
        // WindowEvent::RedrawRequested
        // WindowEvent::KeyboardInput
    }

    fn request_redraw(&mut self) {
        let delta_time = self.last_time.elapsed().as_secs_f32();
        self.last_time = std::time::Instant::now();
        self.angle += delta_time * 10.0;
        if self.angle > 360.0 {
            self.angle -= 360.0;
        }

        let rotation_matrix =
            na::Matrix4::from_axis_angle(&na::Vector3::z_axis(), self.angle.to_radians());

        self.renderer.update_model(rotation_matrix);
        self.renderer.draw();
    }
}
