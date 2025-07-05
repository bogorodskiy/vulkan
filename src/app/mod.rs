mod engine;

use crate::app::engine::{BaseEngine, Engine, UninitializedEngine};
use winit::application::ApplicationHandler;
use winit::event::StartCause;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowId;

pub struct App {
    engine: Box<dyn BaseEngine>,
}

impl App {
    pub fn new() -> Self {
        Self {
            engine: Box::new(UninitializedEngine),
        }
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, start_cause: StartCause) {
        match start_cause {
            StartCause::Init => {
                self.engine.request_redraw();
            }
            // TODO: what is WaitCancelled?
            StartCause::Poll | StartCause::WaitCancelled { .. } => {
                // Regular frame update
                self.engine.request_redraw();
            }
            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.engine = Box::new(Engine::new(event_loop).unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.engine.window_event(event_loop, event);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // TODO: Implement proper suspension handling
    }
}
