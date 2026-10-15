use std::{cell::RefCell,time::Instant};
use serde_json::{Value,json};
thread_local! {
    static ENABLED: RefCell<bool> = const { RefCell::new(false) };
    static EVENTS: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
    static DEVICE: RefCell<Option<burn::tensor::Device>> = const { RefCell::new(None) };
}
pub fn device(d: burn::tensor::Device) { DEVICE.with(|v| *v.borrow_mut()=Some(d)); }
fn sync(){ DEVICE.with(|d| {if let Some(d)=d.borrow().as_ref(){d.sync().expect("GPU synchronization failed");}}); }
pub fn begin(enabled:bool){ENABLED.with(|e| *e.borrow_mut()=enabled);EVENTS.with(|e| e.borrow_mut().clear());}
pub fn take()->Vec<Value>{EVENTS.with(|e| e.borrow().clone())}
pub struct Scope { name: &'static str, start:Instant, enabled:bool, gpu:bool, shape:Vec<usize> }
impl Scope {
    pub fn new(name:&'static str)->Self{let enabled=ENABLED.with(|e| *e.borrow());if enabled{sync();}Self{name,start:Instant::now(),enabled,gpu:true,shape:Vec::new()}}
    pub fn cpu(name:&'static str)->Self{Self{name,start:Instant::now(),enabled:ENABLED.with(|e| *e.borrow()),gpu:false,shape:Vec::new()}}
    pub fn shape(&mut self,shape:Vec<usize>){self.shape=shape;}
}
impl Drop for Scope {fn drop(&mut self){if self.enabled{if self.gpu{sync();}EVENTS.with(|e|e.borrow_mut().push(json!({"stage":self.name,"ms":self.start.elapsed().as_secs_f64()*1000.,"shape":self.shape})));}}}
