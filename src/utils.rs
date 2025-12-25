#![allow(dead_code)]

use ndarray::{Array, ArrayBase, Axis, DataShared, Ix, RemoveAxis, Zip};
use ndarray_rand::rand::{rng, Rng};
use ndarray_rand::rand::prelude::SliceRandom;

pub struct Permutation {
    indices: Vec<Ix>,
}

impl Permutation {
    pub fn random(size: usize) -> Self {
        Self::random_using(&mut rng(), size)
    }

    pub fn random_using(rng: &mut impl Rng, size: usize) -> Self {
        let mut indices = (0..size).collect::<Vec<_>>();
        indices.shuffle(rng);
        Permutation {
            indices
        }
    }

    pub fn ordered(size: usize) -> Self {
        Permutation {
            indices: (0..size).collect()
        }
    }

    fn correct(&self) -> bool {
        let axis_len = self.indices.len();
        let mut seen = vec![false; axis_len];
        for &i in &self.indices {
            match seen.get_mut(i) {
                None => return false,
                Some(s) => {
                    if *s {
                        return false;
                    } else {
                        *s = true;
                    }
                }
            }
        }
        true
    }
}

pub trait PermuteArray {
    type Elem;
    type Dim;

    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<Self::Elem, Self::Dim>;
}

impl<I, A, D> PermuteArray for ArrayBase<I, D>
where
    D: RemoveAxis,
    I: DataShared<Elem = A>,
    A: Clone
{
    type Elem = A;
    type Dim = D;

    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<Self::Elem, Self::Dim> {
        let axis_len = self.len_of(axis);
        assert_eq!(axis_len, perm.indices.len());
        debug_assert!(perm.correct());

        let result = Array::build_uninit(self.dim(), |mut view| {
            let mut moved_elements = 0;
            for &perm_i in perm.indices.iter() {
                Zip::from(view.index_axis_mut(axis, perm_i))
                    .and(self.index_axis(axis, perm_i))
                    .for_each(|to, from| {
                        to.write(from.clone());
                        moved_elements += 1;
                    });
            }
            debug_assert_eq!(view.len(), moved_elements);
        });

        unsafe { result.assume_init() }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array3, Axis};
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;
    use crate::utils::{Permutation, PermuteArray};

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_random_permutation() {
        for _ in 0..1_000_000 {
            let perm = Permutation::random(10);
            assert!(perm.correct());
        }
    }

    #[test]
    fn test_array_permutation() {
        let array = Array3::<f32>::random((10, 20, 30), StandardNormal);
        let perm = Permutation::ordered(20);

        let result = array.view().permute_axis(Axis(1), &perm);
        assert_eq!(result, array);
    }
}
