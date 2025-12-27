#![allow(dead_code)]

use crate::differentiation::{Adr, Record};
use ndarray::{ArrayViewMut, Dimension, LinalgScalar};
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal};
use num_traits::{Float, FromPrimitive, Zero};
use std::sync::Arc;

enum Storage<T: ?Sized + 'static> {
    Boxed(Arc<T>),
    Static(&'static T),
}

pub trait OnceDifferentiableFunctionOps<T> {
    fn function(&self, x: T) -> T;
    fn derivative(&self, x: T) -> T;

    fn ad_function(&self, x: Adr<T>) -> Adr<T>;

    fn into_boxed(self) -> OnceDifferentiableFunction<T>;
}

pub struct OnceDifferentiableFunction<T: 'static> {
    function: Storage<dyn Fn(T) -> T + Send + Sync>,
    derivative: Storage<dyn Fn(T) -> T + Send + Sync>,
    ad_function: Storage<dyn Fn(Adr<T>) -> Adr<T> + Send + Sync>
}

pub trait WeightsInitialization<T> {
    fn distribution(&self, output: usize, input: usize) -> impl Distribution<T>;
}

pub trait ArrayFunction<D: Dimension> {
    fn call<T: LinalgScalar + Exp>(&self, input: ArrayViewMut<T, D>);
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

impl<T: Exp + Clone + Zero> Exp for Record<'_, T> {
    #[inline]
    fn exp(self) -> Self {
        self.unary(T::exp, T::exp)
    }
}

pub struct Softmax;
pub struct Linear;

pub struct Xavier;
pub struct He;
pub struct Standard;

impl<D: Dimension> ArrayFunction<D> for Softmax {
    #[inline]
    fn call<T: LinalgScalar + Exp>(&self, mut input: ArrayViewMut<T, D>) {
        let sum = input.iter().map(|it| it.exp()).fold(T::zero(), T::add);
        input.map_inplace(|x| *x = x.exp() / sum)
    }
}

impl<D: Dimension> ArrayFunction<D> for Linear {
    #[inline]
    fn call<T>(&self, _: ArrayViewMut<T, D>) {}
}

impl<T> WeightsInitialization<T> for Xavier
where
    T: Float + FromPrimitive,
    StandardNormal: Distribution<T>,
{
    fn distribution(&self, output: usize, input: usize) -> impl Distribution<T> {
        let two = T::from_usize(2).unwrap();
        let output = T::from_usize(output).unwrap();
        let input = T::from_usize(input).unwrap();
        Normal::new(T::zero(), (two / (output + input)).sqrt()).unwrap()
    }
}

impl<T> WeightsInitialization<T> for He
where
    T: Float + FromPrimitive,
    StandardNormal: Distribution<T>,
{
    fn distribution(&self, _: usize, input: usize) -> impl Distribution<T> {
        let two = T::from_usize(2).unwrap();
        let input = T::from_usize(input).unwrap();
        Normal::new(- T::one(), (two / input).sqrt()).unwrap()
    }
}

impl<T> WeightsInitialization<T> for Standard
where
    T: Float,
    StandardNormal: Distribution<T>
{
    fn distribution(&self, _: usize, _: usize) -> impl Distribution<T> {
        StandardNormal
    }
}

impl<T: ?Sized + 'static> Clone for Storage<T> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Storage::Boxed(x) => Storage::Boxed(x.clone()),
            Storage::Static(x) => Storage::Static(x),
        }
    }
}

impl<T: 'static> Clone for OnceDifferentiableFunction<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            derivative: self.derivative.clone(),
            ad_function: self.ad_function.clone()
        }
    }
}

impl<T: 'static> OnceDifferentiableFunction<T> {
    pub fn new<F, G, H>(f: F, df: G, adr_f: H) -> Self
    where
        F: Fn(T) -> T + 'static + Send + Sync,
        G: Fn(T) -> T + 'static + Send + Sync,
        H: Fn(Adr<T>) -> Adr<T> + 'static + Send + Sync
    {
        Self {
            function: Storage::Boxed(Arc::new(f)),
            derivative: Storage::Boxed(Arc::new(df)),
            ad_function: Storage::Boxed(Arc::new(adr_f)),
        }
    }

    pub fn from_static<F, G, H>(f: &'static F, df: &'static G, adr_f: &'static H) -> Self
    where
        F: Fn(T) -> T + Send + Sync,
        G: Fn(T) -> T + Send + Sync,
        H: Fn(Adr<T>) -> Adr<T> + Send + Sync
    {
        Self {
            function: Storage::Static(f),
            derivative: Storage::Static(df),
            ad_function: Storage::Static(adr_f)
        }
    }

    pub fn apply<F, G, H>(self, f_map: F, df_map: G, adr_f_map: H) -> OnceDifferentiableFunction<T>
    where
        F: Fn(T) -> T + 'static + Send + Sync,
        G: Fn(T) -> T + 'static + Send + Sync,
        H: Fn(Adr<T>) -> Adr<T> + 'static + Send + Sync,
    {
        OnceDifferentiableFunction::new(
            move |x| {
                f_map(match &self.function {
                    Storage::Boxed(f) => f(x),
                    Storage::Static(f) => f(x),
                })
            },
            move |x| {
                df_map(match &self.derivative {
                    Storage::Boxed(f) => f(x),
                    Storage::Static(f) => f(x),
                })
            },
            move |x| {
                adr_f_map(match &self.ad_function {
                    Storage::Boxed(f) => f(x),
                    Storage::Static(f) => f(x),
                })
            },
        )
    }
}

impl<T> OnceDifferentiableFunctionOps<T> for OnceDifferentiableFunction<T> {
    #[inline(always)]
    fn function(&self, x: T) -> T {
        match &self.function {
            Storage::Boxed(f) => f(x),
            Storage::Static(f) => f(x),
        }
    }

    #[inline(always)]
    fn derivative(&self, x: T) -> T {
        match &self.derivative {
            Storage::Boxed(f) => f(x),
            Storage::Static(f) => f(x),
        }
    }

    #[inline(always)]
    fn ad_function(&self, x: Adr<T>) -> Adr<T> {
        match &self.ad_function {
            Storage::Boxed(f) => f(x),
            Storage::Static(f) => f(x)
        }
    }

    fn into_boxed(self) -> OnceDifferentiableFunction<T>
    {
        self
    }
}
