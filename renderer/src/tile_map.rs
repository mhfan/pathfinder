// pathfinder/renderer/src/tile_map.rs
//
// Copyright © 2019 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use pathfinder_geometry::basic::point::Point2DI32;
use pathfinder_geometry::basic::rect::RectI32;

#[derive(Debug)]
pub struct DenseTileMap<T> {
    pub data: Vec<T>,
    pub rect: RectI32,
}

impl<T> DenseTileMap<T> {
    #[inline]
    pub fn new(rect: RectI32) -> DenseTileMap<T>
    where
        T: Copy + Clone + Default,
    {
        let length = rect.size().x() as usize * rect.size().y() as usize;
        DenseTileMap {
            data: vec![T::default(); length],
            rect,
        }
    }

    #[inline]
    pub fn from_builder<F>(build: F, rect: RectI32) -> DenseTileMap<T>
    where
        F: FnMut(usize) -> T,
    {
        let length = rect.size().x() as usize * rect.size().y() as usize;
        DenseTileMap {
            data: (0..length).map(build).collect(),
            rect,
        }
    }

    #[inline]
    pub fn coords_to_index(&self, coords: Point2DI32) -> Option<usize> {
        // TODO(pcwalton): SIMD?
        if coords.x() < self.rect.min_x()
            || coords.x() >= self.rect.max_x()
            || coords.y() < self.rect.min_y()
            || coords.y() >= self.rect.max_y()
        {
            return None;
        }
        Some(self.coords_to_index_unchecked(coords))
    }

    #[inline]
    pub fn coords_to_index_unchecked(&self, coords: Point2DI32) -> usize {
        (coords.y() - self.rect.min_y()) as usize * self.rect.size().x() as usize
            + (coords.x() - self.rect.min_x()) as usize
    }

    #[inline]
    pub fn index_to_coords(&self, index: usize) -> Point2DI32 {
        let (width, index) = (self.rect.size().x(), index as i32);
        self.rect.origin() + Point2DI32::new(index % width, index / width)
    }
}
