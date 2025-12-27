#![allow(dead_code)]

use crate::differentiation::Adr;
use crate::function_v2::OnceDifferentiableFunction;
use num_traits::{Float, One, Zero};

pub fn sigmoid_fn<T: Float>() -> OnceDifferentiableFunction<T> {
    OnceDifferentiableFunction::from_static(
        &|x: T| T::one() / (T::one() + T::exp(-x)),
        &|x| {
            let z = T::one() / (T::one() + T::exp(-x));
            z * (T::one() - z)
        },
        &|x| Adr::one() / (Adr::one() + (-x).exp()),
    )
}

pub fn linear_fn<T: One>() -> OnceDifferentiableFunction<T> {
    OnceDifferentiableFunction::from_static(&|x: T| x, &|_| T::one(), &|x| x)
}

pub fn relu_fn<T: PartialOrd + Zero + One + Clone>() -> OnceDifferentiableFunction<T> {
    OnceDifferentiableFunction::from_static(
        &|x: T| {
            if x >= T::zero() { x } else { T::zero() }
        },
        &|x| {
            if x >= T::zero() { T::one() } else { T::zero() }
        },
        &|x| {
            if x >= Adr::zero() { x } else { Adr::zero() }
        },
    )
}

pub fn softplus<T: Float>() -> OnceDifferentiableFunction<T> {
    OnceDifferentiableFunction::from_static(
        &|x: T| x.exp().ln_1p(),
        &|x| {
            let exp = x.exp();
            exp / (T::one() + exp)
        },
        &|x| x.exp().ln_1p(),
    )
}

pub fn elu<T: Float + Send + Sync>(alpha: T) -> OnceDifferentiableFunction<T> {
    OnceDifferentiableFunction::new(
        move |x: T| {
            if x > T::zero() { x } else { alpha * x.exp_m1() }
        },
        move |x: T| {
            if x > T::zero() {
                T::one()
            } else {
                alpha * x.exp()
            }
        },
        move |x| {
            if x > Adr::zero() {
                x
            } else {
                Adr::constant(alpha) * x.exp_m1()
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::activation::{linear_fn, relu_fn, sigmoid_fn, softplus};
    use crate::function_v2::OnceDifferentiableFunctionOps;
    use ndarray_linalg::aclose;

    #[test]
    fn test_sigmoid_derivative() {
        let f = sigmoid_fn();
        aclose(f.function(0.1), 0.52497918747894, 1e-15);
        aclose(f.derivative(0.1), 0.24937604019289197, 1e-15);
    }

    #[test]
    fn test_linear_derivative() {
        let f = linear_fn();
        aclose(f.function(0.1), 0.1, 1e-15);
        aclose(f.derivative(0.1), 1.0, 1e-15);
    }

    #[test]
    fn test_relu_derivative() {
        let f = relu_fn();
        aclose(f.function(-0.5), 0.0, 1e-15);
        aclose(f.derivative(-0.5), 0.0, 1e-15);
    }

    #[test]
    fn test_softplus_derivative() {
        let f = softplus();
        aclose(f.function(0.01), 0.6981596805078623, 1e-15);
        aclose(f.derivative(0.01), 0.5024999791668750, 1e-15);
    }
}
