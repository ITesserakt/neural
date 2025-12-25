#![allow(dead_code)]

use crate::function_v2::OnceDifferentiableFunction;
use num_traits::{Float, One};

pub fn sigmoid_fn<'a, T: Float>() -> OnceDifferentiableFunction<T> {
    OnceDifferentiableFunction::from_static(&|x: T| T::one() / (T::one() + T::exp(-x)), &|x| {
        let z = T::one() / (T::one() + T::exp(-x));
        z * (T::one() - z)
    })
}

pub fn linear_fn<'a, T: One>() -> OnceDifferentiableFunction<T> {
    OnceDifferentiableFunction::from_static(&|x: T| x, &|_| T::one())
}

pub fn relu_fn<'a, T: Float>() -> OnceDifferentiableFunction<T> {
    OnceDifferentiableFunction::from_static(&|x: T| x.max(T::zero()), &|x| {
        if x > T::zero() { T::one() } else { T::zero() }
    })
}

#[cfg(test)]
mod tests {
    use crate::activation::{linear_fn, relu_fn, sigmoid_fn};
    use ndarray_linalg::aclose;

    #[test]
    fn test_sigmoid_derivative() {
        let f = sigmoid_fn();
        aclose(f.call(0.1), 0.52497918747894, 1e-15);
        aclose(f.derivative(0.1), 0.24937604019289197, 1e-15);
    }

    #[test]
    fn test_linear_sigmoid() {
        let f = linear_fn();
        aclose(f.call(0.1), 0.1, 1e-15);
        aclose(f.derivative(0.1), 1.0, 1e-15);
    }

    #[test]
    fn test_relu_derivative() {
        let f = relu_fn();
        aclose(f.call(-0.5), 0.0, 1e-15);
        aclose(f.derivative(-0.5), 0.0, 1e-15);
    }
}
