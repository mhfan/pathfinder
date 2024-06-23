/****************************************************************
 * $ID: femtovg.rs      Sat 04 Nov 2023 15:13:31+0800           *
 *                                                              *
 * Maintainer: 范美辉 (MeiHui FAN) <mhfan@ustc.edu>              *
 * Copyright (c) 2024 M.H.Fan, All rights reserved.             *
 ****************************************************************/

use pathfinder_canvas::{Canvas, CanvasRenderingContext2D, Path2D,
    FillStyle, LineJoin, CanvasFontContext, TextAlign, TextBaseline};
use pathfinder_content::{fill::FillRule, gradient::Gradient, stroke::LineCap};
use pathfinder_geometry::{vector::{Vector2F, vec2f, vec2i}, //transform2d::Transform2F,
    rect::RectF, line_segment::LineSegment2F};
use pathfinder_simd::default::F32x2;
use pathfinder_gl::{GLDevice, GLVersion};
use pathfinder_color::{rgbau, rgbf, rgbu};
use pathfinder_renderer::{options::BuildOptions,
    concurrent::{scene_proxy::SceneProxy, rayon::RayonExecutor},
    gpu::{renderer::Renderer, options::{DestFramebuffer, RendererMode, RendererOptions}}};
use pathfinder_resources::{embedded::EmbeddedResourceLoader, fs::FilesystemResourceLoader};

use std::{collections::VecDeque, time::Instant, error::Error, fs, env};
use winit::{application::ApplicationHandler, window::{Window, WindowId},
    event_loop::{ActiveEventLoop, EventLoop}, event::WindowEvent};

#[cfg(not(windows))] #[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[cfg_attr(coverage_nightly, coverage(off))] //#[cfg(not(tarpaulin_include))]
fn main() -> Result<(), Box<dyn Error>> {
    eprintln!("{} v{}-g{}, {} 🦀\n{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"),
        env!("BUILD_GIT_HASH"), env!("BUILD_TIMESTAMP"),
        env!("CARGO_PKG_AUTHORS").replace(':', ", "));
        //build_time::build_time_local!("%H:%M:%S%:z %Y-%m-%d"), //option_env!("ENV_VAR_NAME");
    println!("Usage: {} [<path-to-file>]", env::args().next().unwrap());

    let mut app = WinitApp::new();
    app.load_file(env::args().nth(1).unwrap_or("".to_owned()))?;
    let event_loop = EventLoop::new()?;
    //use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
    //event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;  Ok(())
}

impl ApplicationHandler for WinitApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Err(err) = self.init_state(event_loop, "SVG Renderer - Pathfinder") {
            eprintln!("Failed from surfman: {err:?}");
        }
    }

    fn window_event(&mut self, _loop: &ActiveEventLoop, _wid: WindowId, event: WindowEvent) {
        //if !self.window.as_ref().is_some_and(|window| window.id() == wid) { return }
        use winit::{keyboard::{Key, NamedKey}, event::*};

        match event {   //WindowEvent::Destroyed => dbg!(),
            WindowEvent::CloseRequested => self.exit = true,
            WindowEvent::Focused(bl) => self.focused = bl,

            #[cfg(not(target_arch = "wasm32"))] WindowEvent::Resized(size) => {
                if let Some((device, glctx)) = &mut self.device_glctx {
                    let mut surface = device
                        .unbind_surface_from_context(glctx).unwrap().unwrap();
                    let _ = device.resize_surface(glctx, &mut surface,
                        (size.width as i32, size.height as i32).into());
                    device.bind_surface_to_context(glctx, surface).unwrap();
                }
                if let Some((_, renderer)) = &mut self.scene_render {
                    renderer.options_mut().dest = DestFramebuffer::full_window(
                        vec2i(size.width as _, size.height as _));
                    renderer.dest_framebuffer_size_changed();
                }   self.mouse_pos = Default::default();
                self.resize_viewport(Some((size.width as _, size.height as _)));
            }   // first occur on window creation
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key,
                state: ElementState::Pressed, .. }, .. } => match logical_key.as_ref() {
                Key::Named(NamedKey::Escape) =>     self.exit = true,
                Key::Named(NamedKey::Space)  => {   self.paused = !self.paused;
                                                    self.prevt = Instant::now(); }
                #[cfg(feature =  "lottie")]
                Key::Character(ch) => if self.paused { match ch {
                    "n" | "N" => {  use std::time::Duration;  // XXX:
                        let AnimGraph::Lottie(lottie) =
                            &self.graph else { return };
                        self.prevt = Instant::now() -
                            Duration::from_millis((1000. / lottie.fr) as _);
                        let Some(window) = &self.window else { return };
                        window.request_redraw();
                    }   _ => (),
                } }     _ => (),
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = matches!(state, ElementState::Pressed);
                #[cfg(feature = "rive-rs")]
                if let AnimGraph::Rive((scene, viewport)) = &mut self.graph {
                    match state {
                        ElementState::Pressed  =>
                            scene.pointer_down(self.mouse_pos.x(), self.mouse_pos.y(), viewport),
                        ElementState::Released =>
                            scene.pointer_up  (self.mouse_pos.x(), self.mouse_pos.y(), viewport),
                    }
                }
            }
            WindowEvent::MouseWheel { delta: MouseScrollDelta::LineDelta(_, y), .. } => {
                let Some(ctx2d) = &mut self.ctx2d else { return };
                let origin = ctx2d.transform().inverse() * self.mouse_pos;
                let scale = y / 10. + 1.;       ctx2d.translate( origin);
                ctx2d.scale(vec2f(scale, scale));    ctx2d.translate(-origin);
            }
            WindowEvent::CursorMoved { position, .. } => {
                if  self.dragging {
                    if let Some(ctx2d) = &mut self.ctx2d {
                        let trfm = ctx2d.transform().inverse();
                        let newp   = trfm * vec2f(position.x as _, position.y as _);
                        ctx2d.translate(newp - trfm * self.mouse_pos);
                    }
                }   self.mouse_pos = vec2f(position.x as _, position.y as _);

                #[cfg(feature = "rive-rs")]
                if let AnimGraph::Rive((scene, viewport)) = &mut self.graph {
                    scene.pointer_move(self.mouse_pos.x(), self.mouse_pos.y(), viewport);
                }
            }
            WindowEvent::DroppedFile(path) => {
                self.mouse_pos = Default::default();
                let _ = self.load_file(path);   self.resize_viewport(None);
                let Some(window) = &self.window else { return };
                window.request_redraw();
            }
            WindowEvent::RedrawRequested =>     self.redraw(),
            _ => (),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.exit {    self.window = None;   event_loop.exit(); }    // avoid segmentfault
        if let Some(window)  = &self.window {
            if self.focused && !self.paused { window.request_redraw(); }
        }
    }
}

struct WinitApp {
    exit: bool,
    paused: bool,
    focused: bool,
    dragging: bool,
    mouse_pos: Vector2F,
    start_time: Instant,

    prevt: Instant,
    perf: PerfGraph,
    graph: AnimGraph,

    ctx2d: Option<CanvasRenderingContext2D>,
    scene_render: Option<(SceneProxy, Renderer<GLDevice>)>,
    device_glctx: Option<(surfman::Device, surfman::Context)>,
    window: Option<Window>,
}

#[cfg(feature =  "lottie")] use inlottie::schema::Animation;
#[cfg(feature = "rive-rs")] use inlottie::rive_nvg::RiveNVG;

use demo_nvg::{DemoData, render_demo};
mod demo_nvg;

enum AnimGraph {
    #[cfg(feature =  "lottie")] Lottie(Box<Animation>),
    #[cfg(feature = "rive-rs")]
    Rive((Box<dyn rive_rs::Scene<RiveNVG<OpenGl>>>, rive_rs::Viewport)),
    #[allow(clippy::upper_case_acronyms)] SVG(Box<usvg::Tree>),
    Demo(DemoData),
    None, // for logo/testcase
}

impl Drop for WinitApp {
    fn drop(&mut self) {
        if let Some((device, glctx)) = &mut self.device_glctx {
            device.destroy_context(glctx).unwrap();
        }
    }
}

impl WinitApp {
    fn new() -> Self {
        let res = FilesystemResourceLoader::locate();
        use {pathfinder_resources::ResourceLoader, std::sync::Arc};
        use font_kit::{handle::Handle, sources::mem::MemSource};

        let font_data = vec![   // XXX: greatly improved performance/fps
            Handle::from_memory(Arc::new(res.slurp("fonts/Roboto-Regular.ttf").unwrap()), 0),
            Handle::from_memory(Arc::new(res.slurp("fonts/NotoEmoji-Regular.ttf").unwrap()), 0),
            Handle::from_memory(Arc::new(res.slurp("fonts/Roboto-Bold.ttf").unwrap()), 0),
        ];

        // Initialize font state.
        let font_ctx = CanvasFontContext::new(Arc::new(
            MemSource::from_fonts(font_data.into_iter()).unwrap()));
        //let font_ctx = CanvasFontContext::from_system_source();

        let mut ctx2d = Canvas::new(vec2f(0., 0.))
            .get_context_2d(font_ctx);      // XXX: will be resized latter
        let _ = ctx2d.set_font("Roboto-Regular");

        Self { paused: false, focused: true, dragging: false, exit: false,
            perf: PerfGraph::new(), mouse_pos: Default::default(), prevt: Instant::now(),
            graph: AnimGraph::None, ctx2d: Some(ctx2d), start_time: Instant::now(),
            scene_render: None, device_glctx: None, window: None,
        }
    }

    // https://github.com/rust-windowing/glutin/blob/master/glutin_examples/src/lib.rs
    fn init_state(&mut self, event_loop: &ActiveEventLoop, title: &str) ->
        Result<(), surfman::Error> {
        let mut wsize = event_loop.primary_monitor()
            .map(|monitor| monitor.size()).unwrap();
            //.unwrap_or(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
        wsize.width  /= 2;  wsize.height /= 2;

        let window = event_loop.create_window(Window::default_attributes()
            .with_transparent(true).with_inner_size(wsize)
            .with_title(title)).map_err(|err| { dbg!(err); surfman::Error::Failed })?;

        use surfman::{Connection, ContextAttributeFlags, ContextAttributes,
            GLVersion as GLVersionSM, SurfaceAccess, SurfaceType};

        let connection = Connection::new()?;
        //let connection = surfman::SystemConnection::new()?;
        let mut device = connection.create_device(&connection.create_adapter()?)?;
        //.create_hardware_adapter()? .create_software_adapter()? .create_low_power_adapter()?;

        let wsize = window.inner_size();
        use winit::raw_window_handle::HasWindowHandle;
        let surface_type = SurfaceType::Widget {
            native_widget: connection.create_native_widget_from_window_handle(
                window.window_handle().map_err(|err| {
                    dbg!(err); surfman::Error::Failed })?,
                (wsize.width as i32, wsize.height as i32).into())?
        };

        self.window = Some(window);
        let mut glctx = device.create_context(&device.create_context_descriptor(
            &ContextAttributes { version: GLVersionSM::new(3, 3),
                flags: ContextAttributeFlags::ALPHA, //empty(),
            })?, None)?;

        let surface   = device.create_surface(&glctx, SurfaceAccess::GPUOnly,
            surface_type)?;     // XXX: use SurfaceType::Generic for offscreen
            //SurfaceType::Generic { size: (wsize.width as i32, wsize.height as i32).into() }
            // https://github.com/servo/surfman/blob/main/examples/offscreen.rs

        device.bind_surface_to_context(&mut glctx, surface).map_err(|(err, _)| err)?;
        device.make_context_current(&glctx)?;
        gl::load_with(|symbol_name| device.get_proc_address(&glctx, symbol_name));

        /* let surface = device.create_surface(SurfaceAccess::GPUCPU,
            surface_type)?;     // for macOS SystemConnection

        // XXX: drawing data buffer works like pixels/softbuffer
        device.lock_surface_data(&mut surface)?.data().copy_from_slice(&data);
        device.present_surface  (&mut surface)?; */

        // Create a Pathfinder GL device.
        let gldev = GLDevice::new(GLVersion::GL3,   // XXX: GL4/GLES3?
            device.context_surface_info(&glctx)?.unwrap().framebuffer_object);
        // Get the real size of the window, taking HiDPI into account.

        // Create a Pathfinder renderer.
        let renderer_mode = RendererMode::default_for_device(&gldev);
        let renderer_options = RendererOptions {
            dest: DestFramebuffer::full_window(vec2i(wsize.width as _, wsize.height as _)),
            background_color: Some(rgbf(0.4, 0.4, 0.4)),
            ..RendererOptions::default()
        };

        let renderer = Renderer::new(gldev,
            &EmbeddedResourceLoader::new(), renderer_mode, renderer_options);
        let scene = SceneProxy::new(renderer.mode().level, RayonExecutor);

        self.scene_render = Some((scene, renderer));
        self.device_glctx = Some((device, glctx));

        Ok(())
    }

    fn load_file<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), Box<dyn Error>> {
        let path = path.as_ref();

        //path.rfind('.').map_or("", |i| &path[1 + i..])
        //if fs::metadata(&path).is_ok() {} //if path.exists() {}
        self.graph = match path.extension().and_then(|ext| ext.to_str()) {
            #[cfg(feature =  "lottie")] Some("json") =>
                AnimGraph::Lottie(Box::new(Animation::from_reader(fs::File::open(path)?)?)),

            #[cfg(feature = "rive-rs")] Some("riv")  =>
                AnimGraph::Rive((RiveNVG::new_scene(
                    &fs::read(path)?).unwrap(), Default::default())),

            Some("svg") => {
                let mut usvg_opts = usvg::Options::default();
                        usvg_opts.fontdb_mut().load_system_fonts();
                AnimGraph::SVG(Box::new(usvg::Tree::from_data(&fs::read(path)?, &usvg_opts)?))
            }
            _ => if path.to_str().is_some_and(|path| path.is_empty()) {
                 println!("Load data of NanoVG demo ...");
                AnimGraph::Demo(DemoData::load(&FilesystemResourceLoader::locate()))
            } else {
                eprintln!("Unsupported file format: {}", path.display());
                AnimGraph::None
            }
        };  Ok(())
    }

    fn resize_viewport(&mut self, wsize: Option<(f32, f32)>) {  // maximize & centralize
        let Some(ctx2d) = &mut self.ctx2d else { return };
        let wsize = if let Some(wsize) = wsize {
            ctx2d.canvas_mut().set_size(vec2i(wsize.0 as _, wsize.1 as _));     wsize
        } else {
            let wsize = ctx2d.canvas().size();
            (wsize.x() as _, wsize.y() as _)
        };

        let csize = match &mut self.graph {
            #[cfg(feature =  "lottie")]
            AnimGraph::Lottie(lottie) => (lottie.w as _, lottie.h as _),

            #[cfg(feature = "rive-rs")] AnimGraph::Rive((_, viewport)) => {
                viewport.resize(wsize.0 as _, wsize.1 as _);
                ctx2d.reset_transform();    return
            }
            AnimGraph::SVG(tree) => (tree.size().width(), tree.size().height()),
            AnimGraph::Demo(_) => (demo_nvg::WINDOW_WIDTH as _, demo_nvg::WINDOW_HEIGHT as _),
            AnimGraph::None => { ctx2d.reset_transform();   return }
        };

        ctx2d.reset_transform();
        let scale = (wsize.0 / csize.0).min(wsize.1  / csize.1) * 0.98;     // XXX:
        ctx2d.translate(vec2f((wsize.0 - csize.0 * scale) / 2.,
                              (wsize.1 - csize.1 * scale) / 2.));
        ctx2d.scale(vec2f(scale, scale));
    }

    fn redraw(&mut self) {
        let Some(ctx2d) = &mut self.ctx2d else { return };
        let _elapsed = self.prevt.elapsed();    self.prevt = Instant::now();

        match &mut self.graph {
            #[cfg(feature =  "lottie")] AnimGraph::Lottie(lottie) =>
                if !(lottie.render_next_frame(ctx2d, _elapsed.as_secs_f32())) { return }
                // TODO: draw frame time (lottie.fnth) on screen?

            #[cfg(feature = "rive-rs")]
            AnimGraph::Rive((scene, viewport)) =>
                if !scene.advance_and_maybe_draw(&mut RiveNVG::new(ctx2d),
                    _elapsed, viewport) { return }

            AnimGraph::SVG(tree) =>
                render_nodes(ctx2d, //ctx2d.transform().inverse() *
                    self.mouse_pos, tree.root(), &usvg::Transform::identity()),

            AnimGraph::Demo(data) => {
                let trfm = ctx2d.transform();
                let mouse_pos = trfm.inverse() * self.mouse_pos;

                render_demo(ctx2d, mouse_pos, //ctx2d.canvas().size().to_f32(),
                    vec2f(demo_nvg::WINDOW_WIDTH as _, demo_nvg::WINDOW_HEIGHT as _),
                    self.start_time.elapsed().as_secs_f32(), //trfm.m11(),
                    self.window.as_ref().unwrap().current_monitor()
                        .unwrap().scale_factor() as _, data);
            }
            AnimGraph::None => (),  // XXX: add simple draw test case
        }

        let frame_time = self.prevt.elapsed().as_secs_f32();
        self.perf.render(ctx2d, (3., 3.));     // Render performance graphs.

        let Some((scene, renderer)) =
            &mut self.scene_render else { return };

        // Render the canvas to screen.
        scene.replace_scene(ctx2d.canvas_mut().take_scene());
        //let mut scene = SceneProxy::from_scene(   // XXX: performance downgrade
        //    ctx2d.canvas_mut().take_scene(), renderer.mode().level, RayonExecutor);
        scene.build_and_render(renderer, BuildOptions { //subpixel_aa_enabled: true,
            ..BuildOptions::default() });

        // Add stats to performance graphs.
        if  let Some(gpu_time) = renderer.last_rendering_time() {
            let cpu_time = renderer.stats().cpu_build_time.as_secs_f32();
            let gpu_time = gpu_time.total_time().as_secs_f32();
            self.perf.update(frame_time + cpu_time.max(gpu_time));
        }

        let Some((device, glctx)) =
            &mut self.device_glctx else { return };

        // Present the rendered canvas via `surfman`.
        let mut surface = device
            .unbind_surface_from_context(glctx).unwrap().unwrap();
        device.present_surface(glctx, &mut surface).unwrap();
        device.bind_surface_to_context(glctx, surface).unwrap();
    }
}

pub struct PerfGraph { que: VecDeque<f32>, max: f32, sum: f32/*, time: Instant*/ }

impl PerfGraph { #[allow(clippy::new_without_default)]
    pub fn new() -> Self { Self {
        que: VecDeque::with_capacity(100), max: 0., sum: 0./*, time: Instant::now()*/
    } }

    pub fn update(&mut self, ft: f32) { //debug_assert!(f32::EPSILON < ft);
        //let ft = self.time.elapsed().as_secs_f32();   self.time = Instant::now();
        let fps = 1. / ft;  if self.max <  fps { self.max = fps } // (ft + f32::EPSILON)
        if self.que.len() == 100 {  self.sum -= self.que.pop_front().unwrap_or(0.); }
        self.que.push_back(fps);    self.sum += fps;
    }

    pub fn render(&self, ctx2d: &mut CanvasRenderingContext2D, pos: (f32, f32)) {
        let (rw, rh, mut path) = (100., 20., Path2D::new());
        path.rect(RectF::new(vec2f(0., 0.), vec2f(rw, rh)));

        let last_trfm = ctx2d.transform(); //ctx2d.save();
        ctx2d.reset_transform();     ctx2d.translate(vec2f(pos.0, pos.1));
        ctx2d.set_fill_style(rgbau(0, 0, 0, 99));
        ctx2d.fill_path(path, FillRule::Winding);    // to clear the exact area?
        //ctx2d.fill_rect(RectF::new(vec2f(0., 0.), vec2f(rw, rh)));

        path = Path2D::new();     path.move_to(vec2f(0., rh));
        for i in 0..self.que.len() {  // self.que[i].min(100.) / 100.
            path.line_to(vec2f(rw * i as f32 / self.que.len() as f32,
                rh - rh * self.que[i] / self.max));
        }   path.line_to(vec2f(rw, rh));
        ctx2d.set_fill_style(rgbau(255, 192, 0, 128));
        ctx2d.fill_path(path, FillRule::Winding);

        //let _ = ctx2d.set_font("Roboto-Regular");
        ctx2d.set_fill_style(rgbau(240, 240, 240, 255));
        ctx2d.set_text_baseline(TextBaseline::Top);
        ctx2d.set_text_align(TextAlign::Right);
        ctx2d.set_font_size(14.0); // some fixed values can be moved into the structure

        let fps = self.sum / self.que.len() as f32; // self.que.iter().sum::<f32>()
        ctx2d.fill_text(&format!("{fps:.2} FPS"), vec2f(rw - 10., 2.,));
        ctx2d.reset_transform();    ctx2d.set_transform(&last_trfm);   //ctx2d.restore();
    }
}

#[allow(clippy::only_used_in_recursion)] fn render_nodes(ctx2d: &mut CanvasRenderingContext2D,
    mouse: Vector2F, parent: &usvg::Group, trfm: &usvg::Transform) {
    fn convert_paint(paint: &usvg::Paint, opacity: usvg::Opacity,
        _trfm: &usvg::Transform) -> Option<FillStyle> {
        fn convert_stops(grad: &mut Gradient, stops: &[usvg::Stop], opacity: usvg::Opacity) {
            stops.iter().for_each(|stop| {  let color = stop.color();
                let mut fc = rgbu(color.red, color.green, color.blue);
                fc.a = (stop.opacity() * opacity).get() as u8;
                grad.add_color_stop(fc, stop.offset().get());
            });
        }

        Some(match paint { usvg::Paint::Pattern(_) => { // trfm should be applied here
                eprintln!("Not support pattern painting"); return None }
            // https://github.com/RazrFalcon/resvg/blob/master/crates/resvg/src/path.rs#L179
            usvg::Paint::Color(color) => {
                let mut fc = rgbu(color.red, color.green, color.blue);
                fc.a = (opacity.get() * 255.) as u8;    fc.into()
            }

            usvg::Paint::LinearGradient(grad) => {
                let mut pf_grad = Gradient::linear_from_points(
                    vec2f(grad.x1(), grad.y1()), vec2f(grad.x2(), grad.y2()));
                convert_stops(&mut pf_grad, grad.stops(), opacity);
                //pf_grad.apply_transform(Transform2F::row_major(trfm.sx, trfm.kx,
                //    trfm.ky, trfm.sy, trfm.tx, trfm.ty));
                pf_grad.into()
            }
            usvg::Paint::RadialGradient(grad) => {
                let mut pf_grad = Gradient::radial(LineSegment2F::new(
                    vec2f(grad.fx(), grad.fy()), vec2f(grad.cx(), grad.cy())),
                    F32x2::new((grad.cx() - grad.fx()).hypot(grad.cy() - grad.fy()),
                        grad.r().get()));   // XXX: 1./0.
                convert_stops(&mut pf_grad, grad.stops(), opacity);
                //pf_grad.apply_transform(Transform2F::row_major(trfm.sx, trfm.kx,
                //    trfm.ky, trfm.sy, trfm.tx, trfm.ty));
                pf_grad.into()
            }
        })
    }

    for child in parent.children() { match child {
        usvg::Node::Group(group) =>     // trfm is needed on rendering only
            render_nodes(ctx2d, mouse, group, &trfm.pre_concat(group.transform())),
            // TODO: deal with group.clip_path()/mask()

        usvg::Node::Path(path) => if path.is_visible() {
            let tpath = if trfm.is_identity() { None
            } else { path.data().clone().transform(*trfm) };    // XXX:
            let mut fpath = Path2D::new();

            for seg in tpath.as_ref().unwrap_or(path.data()).segments() {
                use usvg::tiny_skia_path::PathSegment;
                match seg {     PathSegment::Close => fpath.close_path(),
                    PathSegment::MoveTo(pt) => fpath.move_to(vec2f(pt.x, pt.y)),
                    PathSegment::LineTo(pt) => fpath.line_to(vec2f(pt.x, pt.y)),

                    PathSegment::QuadTo(ctrl, end) =>
                        fpath.quadratic_curve_to(vec2f(ctrl.x, ctrl.y), vec2f(end.x, end.y)),
                    PathSegment::CubicTo(ctrl0, ctrl1, end) =>
                        fpath.bezier_curve_to(vec2f(ctrl0.x, ctrl0.y),
                            vec2f(ctrl1.x, ctrl1.y), vec2f(end.x, end.y)),
                }
            }

            let set_fill_style =
                |ctx2d: &mut CanvasRenderingContext2D, fill: &usvg::Fill| {
                if let Some(style) = convert_paint(fill.paint(),
                    fill.opacity(), trfm) { ctx2d.set_fill_style(style); }

                match fill.rule() {
                    usvg::FillRule::NonZero => FillRule::Winding,
                    usvg::FillRule::EvenOdd => FillRule::EvenOdd,
                }
            };

            let set_stroke_style =
                |ctx2d: &mut CanvasRenderingContext2D, stroke: &usvg::Stroke| {
                if let Some(style) = convert_paint(stroke.paint(),
                    stroke.opacity(), trfm) { ctx2d.set_fill_style(style); }

                    ctx2d.set_miter_limit(stroke.miterlimit().get());
                    ctx2d.set_line_width (stroke.width().get());

                    ctx2d.set_line_join(match stroke.linejoin() { usvg::LineJoin::MiterClip |
                        usvg::LineJoin::Miter => LineJoin::Miter,
                        usvg::LineJoin::Round => LineJoin::Round,
                        usvg::LineJoin::Bevel => LineJoin::Bevel,
                    });
                    ctx2d.set_line_cap (match stroke.linecap () {
                        usvg::LineCap::Butt   => LineCap::Butt,
                        usvg::LineCap::Round  => LineCap::Round,
                        usvg::LineCap::Square => LineCap::Square,
                    });
            };

            match path.paint_order() {
                usvg::PaintOrder::FillAndStroke => {
                    if let Some(fill) = path.fill() {
                        let rule = set_fill_style(ctx2d, fill);
                        ctx2d.  fill_path(fpath.clone(), rule);
                    }
                    if let Some(stroke) = path.stroke() {
                        set_stroke_style(ctx2d, stroke);
                        ctx2d.stroke_path(fpath);
                    }
                }
                usvg::PaintOrder::StrokeAndFill => {
                    if let Some(stroke) = path.stroke() {
                        set_stroke_style(ctx2d, stroke);
                        ctx2d.stroke_path(fpath.clone());
                    }
                    if let Some(fill) = path.fill() {
                        let rule = set_fill_style(ctx2d, fill);
                        ctx2d.  fill_path(fpath, rule);
                    }
                }
            }

            /* if fpath.contains_point(mouse.x(), mouse.y(), FillRule::Winding) {   // FIXME:
                ctx2d.set_line_width(1. / ctx2d.transform().m11());
                ctx2d.set_stroke_style(rgbu(32, 240, 32).into());
                ctx2d.stroke_path(fpath);
            } */
        }

        usvg::Node::Image(img) => if img.is_visible() {
            match img.kind() {            usvg::ImageKind::JPEG(_) |
                usvg::ImageKind::PNG(_) | usvg::ImageKind::GIF(_) => todo!(),
                // https://github.com/linebender/vello_svg/blob/main/src/lib.rs#L212
                usvg::ImageKind::SVG(svg) => render_nodes(ctx2d, mouse, svg.root(), trfm),
            }
        }

        usvg::Node::Text(text) => { let group = text.flattened();
            render_nodes(ctx2d, mouse, group, &trfm.pre_concat(group.transform()));
        }
    } }
}

