use ndarray::{Array, ArrayViewMut, Dimension, LinalgScalar};
use num_traits::{Float, Zero};
use std::sync::Arc;
use crate::differentiation::Record;

enum Storage<T: ?Sized + 'static> {
    Boxed(Arc<T>),
    Static(&'static T),
}

pub struct OnceDifferentiableFunction<T: 'static> {
    function: Storage<dyn Fn(T) -> T + Send + Sync>,
    derivative: Storage<dyn Fn(T) -> T + Send + Sync>,
}

pub trait ArrayFunction<D: Dimension> {
    fn call<T: LinalgScalar + Exp>(&self, input: ArrayViewMut<T, D>);
}

pub trait Exp {
    fn exp(self) -> Self;
}

impl<T: Float> Exp for T {
    fn exp(self) -> Self {
        <T as Float>::exp(self)
    }
}

impl<T: Exp + Clone + Zero> Exp for Record<'_, T> {
    fn exp(self) -> Self {
        self.unary(T::exp, T::exp)
    }
}

pub struct Softmax;
pub struct Linear;

impl<D: Dimension> ArrayFunction<D> for Softmax {
    fn call<T: LinalgScalar + Exp>(&self, mut input: ArrayViewMut<T, D>) {
        input.mapv_inplace(T::exp);
        let sum = input.sum();
        input.map_inplace(|x| *x = *x / sum)
    }
}

impl<D: Dimension> ArrayFunction<D> for Linear {
    fn call<T>(&self, _: ArrayViewMut<T, D>) {
    }
}

impl<T: ?Sized + 'static> Clone for Storage<T> {
    fn clone(&self) -> Self {
        match self {
            Storage::Boxed(x) => Storage::Boxed(x.clone()),
            Storage::Static(x) => Storage::Static(x),
        }
    }
}

impl<T: 'static> Clone for OnceDifferentiableFunction<T> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            derivative: self.derivative.clone(),
        }
    }
}

impl<T: 'static> OnceDifferentiableFunction<T> {
    pub fn new<F, G>(f: F, df: G) -> Self
    where
        F: Fn(T) -> T + 'static + Send + Sync,
        G: Fn(T) -> T + 'static + Send + Sync,
    {
        Self {
            function: Storage::Boxed(Arc::new(f)),
            derivative: Storage::Boxed(Arc::new(df)),
        }
    }

    pub fn from_static<F, G>(f: &'static F, df: &'static G) -> Self
    where
        F: Fn(T) -> T + Send + Sync,
        G: Fn(T) -> T + Send + Sync,
    {
        Self {
            function: Storage::Static(f),
            derivative: Storage::Static(df),
        }
    }

    #[inline(always)]
    pub fn call(&self, x: T) -> T {
        match &self.function {
            Storage::Boxed(f) => f(x),
            Storage::Static(f) => f(x),
        }
    }

    #[inline(always)]
    pub fn derivative(&self, x: T) -> T {
        match &self.derivative {
            Storage::Boxed(f) => f(x),
            Storage::Static(f) => f(x),
        }
    }

    pub fn apply<F, G>(self, f_map: F, df_map: G) -> OnceDifferentiableFunction<T>
    where
        F: Fn(T) -> T + 'static + Send + Sync,
        G: Fn(T) -> T + 'static + Send + Sync,
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
        )
    }
}
