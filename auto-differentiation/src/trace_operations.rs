#![allow(clippy::double_parens)]
/*!
* Operator implementations for Traces
*
* These implementations are written here but Rust docs will display them on the
* [Trace] struct page.
*
* Traces of any Numeric type (provided the type also implements the operations by reference
* as described in the [numeric](super::super::numeric) module) implement all the standard
* library traits for addition, subtraction, multiplication and division, so you can
* use the normal `+ - * /` operators as you can with normal number types. As a convenience,
* these operations can also be used with a Trace on the left hand side and a the same type
* that the Trace is generic over on the right hand side, so you can do
*
* ```
* use easy_ml::differentiation::Trace;
* let x: Trace<f32> = Trace::variable(2.0);
* let y: f32 = 2.0;
* let z: Trace<f32> = x * y;
* assert_eq!(z.number, 4.0);
* ```
*
* or more succinctly
*
* ```
* use easy_ml::differentiation::Trace;
* assert_eq!((Trace::variable(2.0) * 2.0).number, 4.0);
* ```
*
* Traces of a [Real] type (provided the type also implements the operations by reference as
* described in the [numeric](super::super::numeric::extra) module) also implement
* all of those extra traits and operations. Note that to use a method defined in a trait
* you have to import the trait as well as have a type that implements it!
*/

use ndarray::ScalarOperand;
use num_traits::real::Real;
use num_traits::{ConstZero, Num, NumCast, One, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use crate::trace::Trace;

/**
 * A trace is displayed by showing its number component.
 */
impl<T: std::fmt::Display> std::fmt::Display for Trace<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.number)
    }
}

impl<T: ScalarOperand> ScalarOperand for Trace<T> {}

impl<T: Clone> Clone for Trace<T> {
    #[inline]
    fn clone(&self) -> Self {
        Trace {
            number: self.number.clone(),
            derivative: self.derivative.clone(),
        }
    }
}

impl<T: Copy> Copy for Trace<T> {}

impl<T> Add<Self> for Trace<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Trace {
            number: self.number.add(rhs.number),
            derivative: self.derivative.add(rhs.derivative),
        }
    }
}

impl<T> Zero for Trace<T>
where
    T: Zero,
{
    #[inline(always)]
    fn zero() -> Self {
        Trace::constant(T::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.number.is_zero()
    }
}

impl<T> Mul<Self> for Trace<T>
where
    T: Mul<Output = T> + Add<Output = T> + Clone,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Trace {
            number: self.number.clone().mul(rhs.number.clone()),
            derivative: self.derivative * rhs.number + self.number * rhs.derivative,
        }
    }
}

impl<T> One for Trace<T>
where
    T: One + Zero + Clone,
{
    #[inline(always)]
    fn one() -> Self {
        Trace::constant(T::one())
    }
}

impl<T: PartialEq> PartialEq for Trace<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl<T: PartialOrd> PartialOrd for Trace<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

impl<T: Sub<Output = T>> Sub<Self> for Trace<T> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Trace {
            number: self.number - rhs.number,
            derivative: self.derivative - rhs.derivative,
        }
    }
}

impl<T> Div<Self> for Trace<T>
where
    T: Div<Output = T> + Clone,
    T: Mul<Output = T>,
    T: Sub<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Trace {
            number: self.number.clone() / rhs.number.clone(),
            derivative: (self.derivative * rhs.number.clone() - self.number * rhs.derivative)
                / (rhs.number.clone() * rhs.number),
        }
    }
}

impl<T: Real> Rem<Self> for Trace<T> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        Trace {
            number: self.number.rem(rhs.number),
            derivative: self.derivative - rhs.derivative * T::round(self.number / rhs.number),
        }
    }
}

impl<T: Real + Zero + One> Num for Trace<T> {
    type FromStrRadixErr = T::FromStrRadixErr;

    #[inline(always)]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Trace::constant(T::from_str_radix(str, radix)?))
    }
}

impl<T: ToPrimitive> ToPrimitive for Trace<T> {
    #[inline(always)]
    fn to_i64(&self) -> Option<i64> {
        Some(self.number.to_i64()?)
    }

    #[inline(always)]
    fn to_u64(&self) -> Option<u64> {
        Some(self.number.to_u64()?)
    }
}

impl<N: NumCast + ConstZero> NumCast for Trace<N> {
    #[inline(always)]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        Some(Trace::constant(N::from(n)?))
    }
}

impl<T: Neg<Output = T>> Neg for Trace<T> {
    type Output = Trace<T>;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Trace {
            number: -self.number,
            derivative: -self.derivative,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use ndarray_linalg::aclose;
    use num_traits::real::Real;
    use std::ops::Mul;
    use crate::trace::Trace;

    fn f<T: Real>(x: T, y: T) -> T {
        x.sin() * y.cos()
    }

    fn g<T: Real>(x: T) -> T {
        (x + T::from(2).unwrap()) * x.ln()
    }

    fn h<T: Mul<Output = T> + Copy>(x: T) -> T {
        x * x * x
    }

    #[test]
    fn test_simple_derivative() {
        let value = Trace::variable(0.4).unary(g, |x| x.ln() + (x + 2.0) / x);
        aclose(value.number, -2.199097756497972, 1e-15);
        aclose(value.derivative, 5.083709268125845, 1e-15);
    }

    #[test]
    fn test_compound_derivative() {
        let value = f(0.4, 0.7);
        aclose(value, 0.2978435767000479, 1e-15);

        let dx = Trace::variable(0.4).binary(
            &Trace::constant(0.7),
            f,
            |x, y| x.cos() * y.cos(),
            |x, y| -y.sin() * x.sin(),
        );
        aclose(dx.number, value, 1e-15);
        aclose(dx.derivative, 0.7044663052755917, 1e-15);

        let dy = Trace::constant(0.4).binary(
            &Trace::variable(0.7),
            f,
            |x, y| x.cos() * y.cos(),
            |x, y| -y.sin() * x.sin(),
        );
        aclose(dy.number, value, 1e-15);
        aclose(dy.derivative, -0.2508701838500143, 1e-15);
    }

    #[test]
    fn test_gradient() {
        let value = f(0.4, 0.7);
        aclose(value, 0.2978435767000479, 1e-15);

        let (v, ds) = Trace::gradient(
            |xs| {
                xs[0].binary(
                    &xs[1],
                    f,
                    |x, y| x.cos() * y.cos(),
                    |x, y| -y.sin() * x.sin(),
                )
            },
            array![0.4, 0.7],
        );
        aclose(value, v, 1e-15);
        aclose(ds[0], 0.7044663052755917, 1e-15);
        aclose(ds[1], -0.2508701838500143, 1e-15);
    }

    #[test]
    fn test_second_order_derivative() {
        let value = h(4.0);
        aclose(value, 64.0, 1e-15);

        let dx = Trace::derivative(h, 4.0);
        aclose(dx, 48.0, 1e-15);

        let ddx = Trace::derivative(h, Trace::variable(4.0));
        assert_eq!(ddx.number, dx);
        aclose(ddx.derivative, 24.0, 1e-15);
    }
}
