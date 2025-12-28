#![allow(dead_code)]

/*!
 * ORIGINAL CODE BELONGS TO: easy-ml: https://github.com/Skeletonxf/easy-ml/blob/master/src/differentiation.rs
 * (Automatic) Differentiation helpers
 *
 * # Automatic Differentiation
 *
 * This module provides structs for performing Forward and Reverse Automatic Differentiation
 *
 * ## Automatic Differentiation is not [Numerical Differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation)
 *
 * You were probably introduced to differentiation as numeric differentiation,
 * ie if you have a function 3x<sup>2</sup> then you can estimate its gradient
 * at some value x by computing 3x<sup>2</sup> and 3(x+h)<sup>2</sup> where h
 * is a very small value. The tangent line these two points create gives you an approximation
 * of the gradient when you calculate (f(x+h) - f(x)) / h. Unfortunately floating
 * point numbers in computers have limited precision, so this method is only approximate
 * and can result in floating point errors. 1 + 1 might equal 2 but as you go smaller
 * 10<sup>-i</sup> + 10<sup>-i</sup> starts to loook rather like 10<sup>-i</sup> as i goes
 * into double digits.
 *
 * ## Automatic Differentiation is not Symbolic Differentiation
 *
 * If you were taught calculus you have probably done plenty of symbolic differentiation
 * by hand. A function 3x<sup>2</sup> can be symbolically differentiated into 6x by applying
 * simple rules to manipulate the algebra. Unfortunately the rules aren't so simple for
 * more complex expressions such as [exponents](https://www.wolframalpha.com/input/?i=d%28x%5Ee%5E2%29%2Fdx),
 * [logs](https://www.wolframalpha.com/input/?i=d%28log%28log%28x%29%29%29%2Fdx) or
 * [trigonometry](https://www.wolframalpha.com/input/?i=d%28sin%28cos%28x%29%29%29%2Fdx).
 * Symbolic differentiation can give you expressions which are just as or more complicated
 * than the original, and doing it by hand can be error prone. Symbolic Differentiation is
 * also tricky to relate to algorithmic computations that use control structures.
 *
 * ## [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
 *
 * Automatic Differentiation computes the derivative of a function without rewriting
 * the function as symbolic differentiation does and without the precision issues of numerical
 * differentiation by splitting the derivative into lots of tiny steps of basic operations
 * like addition and multiplication. These are combined using the chain rule. The downside
 * is more memory is used than in symbolic or numerical differentiation, as derivatives have
 * to be tracked through the computational graph.
 *
 * # Forward Differentiation
 *
 * Forward Differentiation computes all the gradients in a computational graph with respect
 * to an input. For example, if you have a function f(x, y) = 5x<sup>3</sup> - 4x<sup>2</sup> +
 * 10x - y, then for some actual value of x and y you can compute f(x,y) and δf(x,y)/δx
 * together in one forward pass using forward differentiation. You can also make another pass
 * and compute f(x,y) and δf(x,y)/δy for some actual value of x and y. Forward differentiation
 * in this way requires making 2N passes of the function to compute the derivatives of the output
 * with respect to N inputs. However, you do get the gradients for every output in a single pass
 * This is poorly suited to neural nets as they often have a single output(loss)
 * to differentiate many many inputs with respect to.
 *
 * # Reverse Mode Differentiation
 *
 * Reverse Mode Differentiation computes all the gradients in a computational graph for
 * the same output. For example, if you have a function f(x, y) = 5x<sup>3</sup> -
 * 4x<sup>2</sup> + 10x - y, then for some actual value of x and y you can compute f(x,y)
 * and store all the intermediate results. You can then run a backward pass on the output
 * of f(x, y) and obtain δf(x,y)/δx and δf(x,y)/δy for the actual values of x and y in a
 * single pass. The catch is that reverse mode must store as many intermediate values as
 * there are steps in the function which can use much more memory than forward mode.
 * Reverse mode also requires making N backward passes to get the gradients for N different
 * outputs. This is well suited to neural nets because we often have a single output (loss)
 * to differentiate many inputs with respect to. However, reverse mode will be slower than
 * forward mode if the number of inputs is small or there are many outputs.
 *
 * # Usage
 *
 * [See submodule for usage examples](usage)
 *
 * # Further information
 *
 * - [Automatic Differentiation Step by Step](https://medium.com/@marksaroufim/automatic-differentiation-step-by-step-24240f97a6e6)
 * - [Forward Mode Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers)
 * - [Reverse Mode Automatic Differentiation](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
 * - [Automatic Differentiation: The most criminally underused tool in the potential machine learning toolbox?](https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/)
 * - [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
 */

use std::marker::PhantomData;
use std::ops::{AddAssign, Deref, Index, Mul};
use num_traits::{Float, Num, One, Zero};
use object_pool::Reusable;
use crate::record::{FrozenRecord, Record, WengertList};
use crate::trace::Trace;

pub mod record;
mod record_operations;
pub mod trace;
mod trace_operations;

pub struct Derivatives<C> {
    adjoints: C,
}

pub trait Indexed {
    fn index(&self) -> usize;
}

pub struct NoTape<A>(PhantomData<A>);

impl<A> NoTape<A> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<I: Indexed, C, T> Index<&I> for Derivatives<C>
where
    C: Deref<Target: Index<usize, Output = T>>
{
    type Output = T;

    fn index(&self, value: &I) -> &Self::Output {
        &self.adjoints[value.index()]
    }
}

pub trait AD<'a, T>: Num + Copy {
    type Tape: ADTape<T>;
    type Derivatives<'b>;

    fn constant(value: T) -> Self;
    fn variable(value: T, tape: &'a Self::Tape) -> Self;

    fn apply_function(self, function: impl Fn(T) -> T, derivative: impl Fn(T) -> T) -> Self;
    fn with_derivatives<R>(&self, f: impl FnOnce(Derivatives<Self::Derivatives<'_>>) -> R) -> R;
    fn unwrap(self) -> T;
}

pub trait ADTape<T> {
    type AD<'a>;

    fn reset(&self);
}

impl<'a, T> AD<'a, T> for Record<'a, T>
where
    T: Float + 'static + AddAssign,
{
    type Tape = WengertList<T>;
    type Derivatives<'b> = Reusable<'b, Vec<T>>;

    fn constant(value: T) -> Self {
        Record::constant(value)
    }

    fn variable(value: T, tape: &'a WengertList<T>) -> Record<'a, T> {
        Record::variable(value, tape)
    }

    fn apply_function(self, function: impl Fn(T) -> T, derivative: impl Fn(T) -> T) -> Self {
        self.unary(function, derivative)
    }

    fn with_derivatives<R>(&self, f: impl FnOnce(Derivatives<Reusable<Vec<T>>>) -> R) -> R {
        f(self.derivatives())
    }

    fn unwrap(self) -> T {
        self.number
    }
}

impl<'a, T> AD<'a, T> for Trace<T>
where
    T: One + Float + Zero
{
    type Tape = NoTape<Self>;
    type Derivatives<'b> = T;

    #[inline]
    fn constant(value: T) -> Self {
        Self::variable(value)
    }

    #[inline]
    fn variable(value: T, _: &'a Self::Tape) -> Self {
        Self::variable(value)
    }

    #[inline]
    fn apply_function(self, function: impl Fn(T) -> T, derivative: impl Fn(T) -> T) -> Self {
        self.unary(function, derivative)
    }

    fn with_derivatives<R>(&self, f: impl FnOnce(Derivatives<T>) -> R) -> R {
        f(Derivatives { adjoints: self.derivative })
    }

    #[inline]
    fn unwrap(self) -> T {
        self.number
    }
}

impl<T: Copy> ADTape<T> for WengertList<T>
where
    T: 'static
{
    type AD<'a> = Record<'a, T>;

    fn reset(&self) {
        self.clear();
    }
}

impl<T, A> ADTape<T> for NoTape<A> {
    type AD<'a> = A;

    fn reset(&self) {}
}

pub trait Exp {
    fn exp(self) -> Self;
}

impl<T: Float> Exp for T {
    #[inline]
    fn exp(self) -> Self {
        <T as Float>::exp(self)
    }
}

impl<T: Exp + Copy + Zero> Exp for Record<'_, T> {
    #[inline]
    fn exp(self) -> Self {
        self.unary(T::exp, T::exp)
    }
}

impl<T> Exp for Trace<T>
where
    T: Exp + Mul<Output = T> + Clone
{
    #[inline]
    fn exp(self) -> Self {
        self.unary(T::exp, T::exp)
    }
}

#[cfg(test)]
#[should_panic]
#[test]
fn test_record_derivatives_when_no_history() {
    let record = Record::constant(1.0);
    record.derivatives();
}

#[test]
fn test_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<trace::Trace<f64>>();
    assert_sync::<FrozenRecord<f64>>();
}

#[test]
fn test_send() {
    fn assert_send<T: Send>() {}
    assert_send::<trace::Trace<f64>>();
    assert_send::<FrozenRecord<f64>>();
}

const _: () = {
    assert!(size_of::<Record<f64>>() == 24);
    assert!(size_of::<FrozenRecord<f64>>() == 24);
    assert!(align_of::<Record<f64>>() == 8);
    assert!(align_of::<FrozenRecord<f64>>() == 8);
};
