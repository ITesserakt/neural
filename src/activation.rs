use crate::differentiation::Trace;
use crate::function::{Id, OnceDifferentiableFunction};
use ndarray::{Array1, ScalarOperand};
use num_traits::{ConstOne, ConstZero, Float, One, Zero};

pub fn sigmoid_fn<'a, T: Float + ConstOne + ConstZero>() -> OnceDifferentiableFunction<'a, (T,)> {
    OnceDifferentiableFunction::from_ref(
        &|(x,): (T,)| T::one() / (T::one() + T::exp(-x)),
        &|(x,): (Trace<T>,)| Trace::one() / (Trace::one() + Trace::exp(-x)),
    )
}

pub fn linear_fn<'a, T>() -> OnceDifferentiableFunction<'a, (T,)> {
    OnceDifferentiableFunction::from_ref(&|(x,): (T,)| x, &|(x,): (Trace<T>,)| x)
}

pub fn relu_fn<'a, T: Float + ConstZero + ConstOne>() -> OnceDifferentiableFunction<'a, (T,)> {
    OnceDifferentiableFunction::from_ref(&|(x,): (T,)| x.max(T::zero()), &|(x,): (Trace<T>,)| {
        x.max(Trace::zero())
    })
}

pub fn softmax_fn<'a, T: Float + ScalarOperand + ConstOne + ConstZero>()
-> OnceDifferentiableFunction<'a, Array1<T>> {
    OnceDifferentiableFunction::from_ref(
        &|xs: Array1<T>| {
            let exponentiation = xs.mapv(T::exp);
            let sum = exponentiation.sum();
            exponentiation / sum
        },
        &|xs| {
            let exponentiation = xs.mapv(Trace::exp);
            let sum = exponentiation.sum();
            exponentiation / sum
        },
    )
}

pub fn classify_fn<'a, T: Float + ScalarOperand + ConstOne + ConstZero>()
-> OnceDifferentiableFunction<'a, Array1<T>, Id<T>> {
    softmax_fn::<'a, T>().map(
        move |xs| xs.fold(T::min_value(), |acc, next| acc.max(*next)),
        move |xs| xs.fold(Trace::min_value(), |acc, next| acc.max(*next)),
    )
}

#[cfg(test)]
mod tests {
    use crate::activation::{classify_fn, linear_fn, relu_fn, sigmoid_fn, softmax_fn};
    use ndarray::array;
    use ndarray_linalg::aclose;

    #[test]
    fn test_sigmoid_derivative() {
        let d = sigmoid_fn().derivative(0.1);
        aclose(d.number, 0.52497918747894, 1e-15);
        aclose(d.derivative, 0.24937604019289197, 1e-15);
    }

    #[test]
    fn test_linear_sigmoid() {
        let d = linear_fn().derivative(0.1);
        aclose(d.number, 0.1, 1e-15);
        aclose(d.derivative, 1.0, 1e-15);
    }

    #[test]
    fn test_relu_derivative() {
        let d = relu_fn().derivative(-0.5);
        aclose(d.number, 0.0, 1e-15);
        aclose(d.derivative, 0.0, 1e-15);
    }

    #[test]
    fn test_softmax_derivative() {
        let (v, d) = softmax_fn().jacobian(array![0.3, 0.5, 1.0]);

        assert_eq!(
            v,
            array![0.23611884100011252, 0.28839620365112, 0.4754849553487675]
        );
        assert_eq!(
            d,
            array![
                [
                    0.18036673392487607,
                    -0.06809577735493487,
                    -0.11227095656994122
                ],
                [
                    -0.06809577735493487,
                    0.20522383337074168,
                    -0.13712805601580683
                ],
                [
                    -0.11227095656994122,
                    -0.13712805601580683,
                    0.2493990125857481
                ]
            ]
        );
    }

    #[test]
    fn test_classify_gradient() {
        let (v, d) = classify_fn().gradient(array![0.3, 0.5, 1.0]);
        assert_eq!(v, 0.4754849553487675);

        assert_eq!(d, array![-0.11227095656994122, -0.13712805601580683, 0.2493990125857481]);
    }
}
