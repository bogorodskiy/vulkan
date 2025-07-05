mod renderer;

use crate::app::engine::renderer::VulkanRenderer;
use anyhow::Result;
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
}

impl Engine {
    pub(crate) fn new(event_loop: &ActiveEventLoop) -> Result<Self> {
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan Demo")
            .with_inner_size(LogicalSize::new(800.0, 600.0))
            .with_resizable(false);
        let window = Arc::new(event_loop.create_window(window_attributes)?);
        let renderer = VulkanRenderer::new(window)?;

        Ok(Self { renderer })
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
        self.renderer.draw();
    }
}
