use std::ops::{Add, Mul};
use ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix1};
use num_traits::{One, Zero};

/**
 * A dual number which traces a real number and keeps track of its derivative.
 * This is used to perform Forward Automatic Differentiation
 *
 * Trace implements only first order differentiation. For example, given a function
 * 3x<sup>2</sup>, you can use calculus to work out that its derivative with respect
 * to x is 6x. You can also take the derivative of 6x with respect to x and work out
 * that the second derivative is 6. By instead writing the function 3x<sup>2</sup> in
 * code using Trace types as your numbers you can compute the first order derivative
 * for a given value of x by passing your function `Trace { number: x, derivative: 1.0 }`.
 *
 * ```rust
 * use crate::differentiation::Trace;
 * let x = Trace { number: 3.2, derivative: 1.0 };
 * let dx = Trace::constant(3.0) * x * x;
 * assert_eq!(dx.derivative, 3.2 * 6.0);
 * ```
 *
 * Why the one for the starting derivative? Because δx/δx = 1, as with symbolic
 * differentiation.
 *
 * # Acknowledgments
 *
 * The wikipedia page on [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
 * provided a very useful overview and explanation for understanding Forward Mode Automatic
 * Differentiation as well as the implementation rules.
 */
#[derive(Debug)]
pub struct Trace<T> {
    /**
     * The real number
     */
    pub number: T,
    /**
     * The first order derivative of this number.
     */
    // If we loosen this type from T to a tensor of T of some const-generic
    // dimensionality then we can calculate higher order derivatives with a single
    // Trace type.
    // However, Trace<Trace<f64>> can do such a calculation already for 2nd order
    // (and so on) and requires far less complexity in the API so this might not
    // be that worthwhile. Tensor<T, 1> introduces a lot of boxing that might also
    // hurt first order performance.
    pub derivative: T,
}

/**
 * The main set of methods for using Trace types for Forward Differentiation.
 *
 * The general steps are
 * 1. create one variable
 * 2. create as many constants as needed
 * 3. do operations on the variable and constants
 * 4. the outputs will have derivatives computed which can be accessed from
 * the `.derivative` field, with each derivative being the output with respect
 * to the input variable.
 * 5. if you need derivatives for a different input then do everything all over again
 * or do them all in parallel
 */
impl<T> Trace<T> {
    /**
     * Constants are lifted to Traces with a derivative of 0
     *
     * Why zero for the starting derivative? Because for any constant C
     * δC/δx = 0, as with symbolic differentiation.
     */
    #[inline(always)]
    pub fn constant(c: T) -> Trace<T>
    where
        T: Zero,
    {
        Trace {
            number: c,
            derivative: T::zero()
        }
    }

    /**
     * To lift a variable that you want to find the derivative of
     * a function to, the Trace starts with a derivative of 1
     *
     * Why the one for the starting derivative? Because δx/δx = 1, as with symbolic
     * differentiation.
     */
    #[inline(always)]
    pub fn variable(x: T) -> Trace<T>
    where
        T: One,
    {
        Trace {
            number: x,
            derivative: T::one(),
        }
    }

    /**
     * Computes the derivative of a function with respect to its input x.
     *
     * This is a shorthand for `(function(Trace::variable(x))).derivative`
     *
     * In the more general case, if you provide a function with an input x
     * and it returns N outputs y<sub>1</sub> to y<sub>N</sub> then you
     * have computed all the derivatives δy<sub>i</sub>/δx for i = 1 to N.
     */
    #[inline]
    pub fn derivative(function: impl FnOnce(Trace<T>) -> Trace<T>, x: T) -> T
    where
        T: One,
    {
        function(Trace::variable(x)).derivative
    }

    #[inline]
    pub fn gradient(
        mut function: impl FnMut(ArrayView1<Trace<T>>) -> Trace<T>,
        xs: ArrayBase<impl Data<Elem = T>, Ix1>,
    ) -> (T, Array1<T>)
    where
        T: Zero + Clone + One + PartialEq,
    {
        let mut traces = xs.mapv(Trace::constant);
        let mut result = xs.to_owned();
        let mut value = T::zero();

        let mut prev = None;
        for i in 0..traces.len() {
            traces[i].derivative = T::one();
            if let Some(j) = prev {
                traces[j].derivative = T::zero();
            }
            prev = Some(i);
            let d = function(traces.view());
            value = d.number;
            result[i] = d.derivative;
        }

        (value, result)
    }
}

impl<T> Trace<T>
where
    T: Clone,
{
    /**
     * Creates a new Trace from a reference to an existing Trace by applying
     * some unary function to it which operates on the type the Trace wraps.
     *
     * To compute the new trace, the unary function of some input x to some
     * output y is needed along with its derivative with respect to its input x.
     *
     * For example, tanh is a commonly used activation function, but the Real trait
     * does not include this operation and Trace has no operations for it specifically.
     * However, you can use this function to compute the tanh of a Trace like so:
     *
     * ```
     * use easy_ml::differentiation::Trace;
     * let x = Trace::variable(0.7f32);
     * // the derivative of tanh(x) is sech(x) * sech(x) which is equivalent to
     * // 1 / (cosh(x) * cosh(x))
     * let y = x.unary(|x| x.tanh(), |x| 1.0 / (x.cosh() * x.cosh()));
     * assert_eq!(y.derivative, 1.0f32 / (0.7f32.cosh() * 0.7f32.cosh()));
     * ```
     */
    #[inline]
    pub fn unary(&self, fx: impl Fn(T) -> T, dfx_dx: impl Fn(T) -> T) -> Trace<T>
    where
        T: Mul<Output = T>,
    {
        Trace {
            number: fx(self.number.clone()),
            derivative: self.derivative.clone() * dfx_dx(self.number.clone()),
        }
    }

    /**
     * Creates a new Trace from a reference to two existing Traces by applying
     * some binary function to them which operates on two arguments of the type
     * the Traces wrap.
     *
     * To compute the new trace, the binary function of some inputs x and y to some
     * output z is needed along with its derivative with respect to its first input x and
     * its derivative with respect to its second input y.
     *
     * For example, atan2 takes two arguments, but the Real trait
     * does not include this operation and Trace has no operations for it specifically.
     * However, you can use this function to compute the atan2 of two Traces like so:
     *
     * ```
     * use easy_ml::differentiation::Trace;
     * let x = Trace::variable(3.0f32);
     * let y = Trace::variable(3.0f32);
     * // the derivative of atan2 with respect to x is y/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdx
     * // the derivative of atan2 with respect to y is -x/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdy
     * let z = x.binary(&y,
     *     |x, y| x.atan2(y),
     *     |x, y| y/((x*x) + (y*y)),
     *     |x, y| -x/((x*x) + (y*y))
     * );
     * ```
     */
    #[inline]
    pub fn binary(
        &self,
        rhs: &Trace<T>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) -> Trace<T>
    where
        T: Mul<Output = T>,
        T: Add<Output = T>,
    {
        Trace {
            number: fxy(self.number.clone(), rhs.number.clone()),
            derivative: (self.derivative.clone()
                * dfxy_dx(self.number.clone(), rhs.number.clone()))
                + (rhs.derivative.clone() * dfxy_dy(self.number.clone(), rhs.number.clone())),
        }
    }
}