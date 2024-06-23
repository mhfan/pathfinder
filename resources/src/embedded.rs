// pathfinder/resources/src/embedded.rs
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Embeds needed resources statically in the binary.

use crate::ResourceLoader;
use std::io::{Error as IOError, ErrorKind};

include!(concat!(env!("OUT_DIR"), "/manifest.rs"));

pub struct EmbeddedResourceLoader;

impl EmbeddedResourceLoader {
    #[inline] #[allow(clippy::new_without_default)]
    pub fn new() -> EmbeddedResourceLoader {
        EmbeddedResourceLoader
    }
}

impl ResourceLoader for EmbeddedResourceLoader {
    fn slurp(&self, virtual_path: &str) -> Result<Vec<u8>, IOError> {
        match RESOURCES.iter().find(|&(path, _)| *path == virtual_path) {
            Some((_, data)) => Ok(data.to_vec()),
            None => Err(IOError::from(ErrorKind::NotFound)),
        }
    }
}
