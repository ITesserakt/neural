use std::sync::Arc;
use crate::differentiation::Trace;
use ndarray::{Array, Array1, Array2, Dimension};
use ndarray_linalg::{aclose, Scalar};
use num_traits::{ConstOne, ConstZero, Float};

pub struct Id<T>(pub T);

pub trait Container {
    type T;
    type Unwrap;
    type Map<U>;
}

impl<T> Container for Id<T> {
    type T = T;
    type Unwrap = T;
    type Map<U> = U;
}

impl<T, D> Container for Array<T, D>
where
    D: Dimension,
{
    type T = T;
    type Unwrap = Self;
    type Map<U> = Array<U, D>;
}

pub trait Args {
    type T;
    type Container: Container;
    type Map<U>: Args;
}

impl<A> Args for (A,) {
    type T = A;
    type Container = Id<A>;
    type Map<U> = (U,);
}

impl<A> Args for (A, A) {
    type T = A;
    type Container = Id<A>;
    type Map<U> = (U, U);
}

impl<'a, A, D> Args for Array<A, D>
where
    D: Dimension,
{
    type T = A;
    type Container = Self;
    type Map<U> = Array<U, D>;
}

#[derive(Debug)]
enum Storage<'a, T: ?Sized + 'a> {
    Boxed(Arc<T>),
    Static(&'a T),
}

type RemapArgs<L> = <L as Args>::Map<Trace<<L as Args>::T>>;
type RemapRet<R> = <R as Container>::Map<Trace<<R as Container>::T>>;

pub struct OnceDifferentiableFunction<'a, L, R = <L as Args>::Container>
where
    L: Args + 'a,
    R: 'a + Container,
{
    value: Storage<'a, dyn Fn(L) -> R::Unwrap + Send + Sync + 'a>,
    with_derivative: Storage<'a, dyn Fn(RemapArgs<L>) -> RemapRet<R> + Send + Sync>,
}

impl<'a, T: ?Sized> Clone for Storage<'a, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        match self {
            Storage::Boxed(x) => Self::Boxed(x.clone()),
            Storage::Static(x) => Self::Static(x)
        }
    }
}

impl<'a, L: Args, R: Container> Clone for OnceDifferentiableFunction<'a, L, R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            with_derivative: self.with_derivative.clone()
        }
    }
}

impl<'a, T, R: Container> OnceDifferentiableFunction<'a, (T,), R> {
    #[inline(always)]
    pub fn f1(
        f: impl Fn(T) -> R::Unwrap + Send + Sync + 'a,
        fd: impl Fn(Trace<T>) -> RemapRet<R> + Send + Sync + 'a,
    ) -> Self {
        Self {
            value: Storage::Boxed(Arc::new(move |(x,)| f(x))),
            with_derivative: Storage::Boxed(Arc::new(move |(x,)| fd(x))),
        }
    }

    #[inline(always)]
    pub fn derivative(&self, x: T) -> RemapRet<R>
    where
        T: ConstOne,
    {
        match &self.with_derivative {
            Storage::Boxed(f) => f((Trace::variable(x),)),
            Storage::Static(f) => f((Trace::variable(x),)),
        }
    }
}

impl<'a, T> OnceDifferentiableFunction<'a, Array1<T>, Id<T>> {
    #[inline(always)]
    pub fn gradient(&self, xs: Array1<T>) -> (T, Array1<T>)
    where
        T: ConstZero + ConstOne + Scalar,
    {
        let mut traces = xs.mapv(Trace::constant);
        let mut result = xs.to_owned();
        let mut value = T::zero();
        let mut first = true;

        let mut prev = None;
        for i in 0..traces.len() {
            traces[i].derivative = T::one();
            if let Some(j) = prev {
                traces[j].derivative = T::zero();
            }
            prev = Some(i);
            let d = self.call_with_derivative(traces.to_owned());
            result[i] = d.derivative;

            #[cfg(debug_assertions)]
            {
                if first {
                    first = false;
                } else {
                    aclose(value, d.number, <T::Real as Float>::epsilon());
                }
            }

            value = d.number;
        }

        (value, result)
    }
}

impl<'a, T> OnceDifferentiableFunction<'a, Array1<T>> {
    #[inline(always)]
    pub fn jacobian(&self, xs: Array1<T>) -> (Array1<T>, Array2<T>)
    where
        T: ConstZero + ConstOne + Clone,
    {
        let mut traces = xs.mapv(Trace::constant);
        let mut result = Array2::zeros((xs.dim(), 0));
        let mut value = Array1::zeros(0);

        let mut prev = None;
        for i in 0..traces.len() {
            traces[i].derivative = T::one();
            if let Some(j) = prev {
                traces[j].derivative = T::zero();
            }
            prev = Some(i);
            let d = self.call_with_derivative(traces.to_owned());
            result.push_column(d.mapv(|it| it.derivative).view()).unwrap();

            value = d.mapv(|it| it.number);
        }

        (value, result)
    }
}

impl<'a, L: 'a + Args, R: Container> OnceDifferentiableFunction<'a, L, R> {
    #[inline(always)]
    pub fn new(
        f: impl Fn(L) -> R::Unwrap + Send + Sync + 'a,
        fd: impl Fn(RemapArgs<L>) -> RemapRet<R> + Send + Sync + 'a,
    ) -> Self {
        Self {
            value: Storage::Boxed(Arc::new(f)),
            with_derivative: Storage::Boxed(Arc::new(fd)),
        }
    }

    #[inline(always)]
    pub const fn from_ref(
        f: &'a (impl Fn(L) -> R::Unwrap + Send + Sync),
        fd: &'a (impl Fn(RemapArgs<L>) -> RemapRet<R> + Send + Sync),
    ) -> Self {
        Self {
            value: Storage::Static(f),
            with_derivative: Storage::Static(fd),
        }
    }

    #[inline(always)]
    pub fn call(&self, args: L) -> R::Unwrap {
        match &self.value {
            Storage::Boxed(f) => f(args),
            Storage::Static(f) => f(args),
        }
    }

    #[inline(always)]
    pub fn call_with_derivative(&self, args: RemapArgs<L>) -> RemapRet<R> {
        match &self.with_derivative {
            Storage::Boxed(f) => f(args),
            Storage::Static(f) => f(args),
        }
    }

    #[inline(always)]
    pub fn map<U: Container>(
        self,
        map_f: impl Fn(R::Unwrap) -> U::Unwrap + Send + Sync + 'a,
        map_df: impl Fn(RemapRet<R>) -> RemapRet<U> + Send + Sync + 'a
    ) -> OnceDifferentiableFunction<'a, L, U> {
        let a = self.clone();

        OnceDifferentiableFunction::new(
            move |args| map_f(a.call(args)),
            move |args| map_df(self.call_with_derivative(args))
        )
    }
}
