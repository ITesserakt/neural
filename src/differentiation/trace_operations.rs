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

use crate::differentiation::Trace;
use ndarray::ScalarOperand;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num_traits::float::FloatCore;
use num_traits::real::Real;
use num_traits::{ConstOne, ConstZero, Float, Num, NumCast, One, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::num::FpCategory;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/**
 * A trace is displayed by showing its number component.
 */
impl<T: std::fmt::Display> std::fmt::Display for Trace<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.number)
    }
}

impl<T> Distribution<Trace<T>> for StandardNormal
where
    Self: Distribution<T>,
    T: ConstZero,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Trace<T> {
        Trace::constant(<Self as Distribution<T>>::sample(self, rng))
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
    T: ConstZero,
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

impl<T: ConstZero> ConstZero for Trace<T> {
    const ZERO: Self = Trace::constant(T::ZERO);
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
    T: ConstOne + ConstZero + Clone,
{
    #[inline(always)]
    fn one() -> Self {
        Trace::constant(T::one())
    }
}

impl<T: ConstZero + ConstOne + Clone> ConstOne for Trace<T> {
    const ONE: Self = Trace::constant(T::ONE);
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

impl<T: Real + ConstZero + ConstOne> Num for Trace<T> {
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

impl<T: Real + FloatCore + ConstZero + ConstOne> FloatCore for Trace<T> {
    #[inline(always)]
    fn infinity() -> Self {
        Trace::constant(T::infinity())
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        Trace::constant(T::neg_infinity())
    }

    #[inline(always)]
    fn nan() -> Self {
        Trace::constant(T::nan())
    }

    #[inline(always)]
    fn neg_zero() -> Self {
        Trace::constant(T::neg_zero())
    }

    #[inline(always)]
    fn min_value() -> Self {
        Trace::constant(FloatCore::min_value())
    }

    #[inline(always)]
    fn min_positive_value() -> Self {
        Trace::constant(FloatCore::min_positive_value())
    }

    #[inline(always)]
    fn epsilon() -> Self {
        Trace::constant(FloatCore::epsilon())
    }

    #[inline(always)]
    fn max_value() -> Self {
        Trace::constant(FloatCore::max_value())
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        FloatCore::is_nan(self.number)
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        FloatCore::is_infinite(self.number)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        FloatCore::is_finite(self.number)
    }

    #[inline(always)]
    fn is_normal(self) -> bool {
        FloatCore::is_normal(self.number)
    }

    #[inline(always)]
    fn is_subnormal(self) -> bool {
        FloatCore::is_subnormal(self.number)
    }

    #[inline(always)]
    fn classify(self) -> FpCategory {
        self.number.classify()
    }

    #[inline(always)]
    fn floor(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn round(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn fract(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn signum(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn is_sign_positive(self) -> bool {
        todo!()
    }

    #[inline(always)]
    fn is_sign_negative(self) -> bool {
        todo!()
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn recip(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn powi(self, exp: i32) -> Self {
        todo!()
    }

    #[inline(always)]
    fn to_degrees(self) -> Self {
        Trace::constant(FloatCore::to_degrees(self.number))
    }

    #[inline(always)]
    fn to_radians(self) -> Self {
        Trace::constant(FloatCore::to_radians(self.number))
    }

    #[inline(always)]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.number.integer_decode()
    }
}

impl<T: Float + ConstZero + ConstOne> Float for Trace<T> {
    #[inline(always)]
    fn nan() -> Self {
        Trace::constant(Float::nan())
    }

    #[inline(always)]
    fn infinity() -> Self {
        Trace::constant(Float::infinity())
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        Trace::constant(Float::neg_infinity())
    }

    #[inline(always)]
    fn neg_zero() -> Self {
        Trace::constant(Float::neg_zero())
    }

    #[inline(always)]
    fn min_value() -> Self {
        Trace::constant(Float::min_value())
    }

    #[inline(always)]
    fn min_positive_value() -> Self {
        Trace::constant(Float::min_positive_value())
    }

    #[inline(always)]
    fn max_value() -> Self {
        Trace::constant(Float::max_value())
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        Float::is_nan(self.number)
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        Float::is_infinite(self.number)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        Float::is_finite(self.number)
    }

    #[inline(always)]
    fn is_normal(self) -> bool {
        Float::is_normal(self.number)
    }

    #[inline(always)]
    fn classify(self) -> FpCategory {
        Float::classify(self.number)
    }

    #[inline(always)]
    fn floor(self) -> Self {
        Trace::constant(Float::floor(self.number))
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        Trace::constant(Float::ceil(self.number))
    }

    #[inline(always)]
    fn round(self) -> Self {
        Trace::constant(Float::round(self.number))
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        Trace::constant(Float::trunc(self.number))
    }

    #[inline(always)]
    fn fract(self) -> Self {
        Trace::variable(Float::fract(self.number))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Trace {
            number: Float::abs(self.number),
            derivative: Float::signum(self.number) * self.derivative,
        }
    }

    #[inline(always)]
    fn signum(self) -> Self {
        Trace {
            number: Float::signum(self.number),
            derivative: T::from(2).unwrap() * self.derivative
        }
    }

    #[inline(always)]
    fn is_sign_positive(self) -> bool {
        Float::is_sign_positive(self.number)
    }

    #[inline(always)]
    fn is_sign_negative(self) -> bool {
        Float::is_sign_negative(self.number)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Trace {
            number: Float::mul_add(self.number, a.number, b.number),
            // u'v + uv' + b'
            derivative: Float::mul_add(
                self.derivative,
                a.number,
                Float::mul_add(self.number, a.derivative, b.derivative),
            ),
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Trace {
            number: Float::recip(self.number),
            derivative: -(self.derivative / self.number / self.number),
        }
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        Trace {
            number: Float::powi(self.number, n),
            derivative: { T::from(n).unwrap() * Float::powi(self.number, n - 1) * self.derivative },
        }
    }

    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        let number = Float::powf(self.number, n.number);
        Trace {
            number,
            derivative: number
                * (self.derivative * n.number / self.number
                    + n.derivative * Float::ln(self.number)),
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn exp(self) -> Self {
        let number = Float::exp(self.number);
        Self {
            number,
            derivative: number * self.derivative,
        }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        let number = Float::exp2(self.number);
        Self {
            number,
            derivative: Float::ln(T::from(2).unwrap()) * number * self.derivative,
        }
    }

    #[inline(always)]
    fn ln(self) -> Self {
        Trace {
            number: Float::ln(self.number),
            derivative: self.derivative / self.number,
        }
    }

    #[inline(always)]
    fn log(self, base: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn log2(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn log10(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Trace {
            number: Float::max(self.number, other.number),
            derivative: if self.number >= other.number {
                self.derivative
            } else {
                other.derivative
            },
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn abs_sub(self, other: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn cbrt(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn sin(self) -> Self {
        Trace {
            number: Float::sin(self.number),
            derivative: self.derivative * Float::cos(self.number),
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        Trace {
            number: Float::cos(self.number),
            derivative: -self.derivative * Float::sin(self.number),
        }
    }

    #[inline(always)]
    fn tan(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn asin(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn acos(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn atan(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        todo!()
    }

    #[inline(always)]
    fn exp_m1(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn ln_1p(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn integer_decode(self) -> (u64, i16, i8) {
        Float::integer_decode(self.number)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::differentiation::Trace;
    use ndarray_linalg::aclose;
    use num_traits::real::Real;

    fn f<T: Real>(x: T, y: T) -> T {
        x.sin() * y.cos()
    }

    fn g<T: Real>(x: T) -> T {
        (x + T::from(2).unwrap()) * x.ln()
    }

    fn h<T: Real>(x: T) -> T {
        x.powf(T::from(3).unwrap())
    }

    #[test]
    fn test_simple_derivative() {
        let value = g(0.4);
        aclose(value, -2.199097756497972, 1e-15);

        let derivative = Trace::derivative(g, 0.4);
        aclose(derivative, 5.083709268125845, 1e-15);
    }

    #[test]
    fn test_compound_derivative() {
        let value = f(0.4, 0.7);
        aclose(value, 0.2978435767000479, 1e-15);

        let dx = f(Trace::variable(0.4), Trace::constant(0.7));
        assert_eq!(dx.number, value);
        aclose(dx.derivative, 0.7044663052755917, 1e-15);

        let dy = f(Trace::constant(0.4), Trace::variable(0.7));
        assert_eq!(dy.number, value);
        aclose(dy.derivative, -0.2508701838500143, 1e-15);
    }
    
    #[test]
    fn test_gradient() {
        let value = f(0.4, 0.7);
        aclose(value, 0.2978435767000479, 1e-15);
        
        let (v, ds) = Trace::gradient(|xs| f(xs[0], xs[1]), array![0.4, 0.7]);
        assert_eq!(value, v);
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
