use std::marker::PhantomData;
use crate::function_v2::{OnceDifferentiableFunctionOps, WeightsInitialization, Xavier};

pub struct Hidden<R> {
    pub(super) rng: R,
}
pub struct Ready<F> {
    pub(super) output: F,
}

pub struct LayerConfig<T, F, W>
where
    F: OnceDifferentiableFunctionOps<T>,
    W: WeightsInitialization<T>
{
    pub(super) activation: F,
    pub(super) weights_initialization: W,
    _phantom: PhantomData<T>
}

pub trait IntoLayerConfig<T, F, W>
where 
    F: OnceDifferentiableFunctionOps<T>,
    W: WeightsInitialization<T>
{
    fn into_config(self) -> LayerConfig<T, F, W>;
}

impl<F, T> IntoLayerConfig<T, F, Xavier> for F
where 
    F: OnceDifferentiableFunctionOps<T>,
    Xavier: WeightsInitialization<T>
{
    fn into_config(self) -> LayerConfig<T, F, Xavier> {
        LayerConfig {
            activation: self,
            weights_initialization: Xavier,
            _phantom: PhantomData,
        }
    }
}

impl<F, T, W> IntoLayerConfig<T, F, W> for (F, W)
where 
    F: OnceDifferentiableFunctionOps<T>,
    W: WeightsInitialization<T>
{
    fn into_config(self) -> LayerConfig<T, F, W> {
        LayerConfig {
            activation: self.0,
            weights_initialization: self.1,
            _phantom: PhantomData,
        }
    }
}

impl<F, T, W> IntoLayerConfig<T, F, W> for LayerConfig<T, F, W>
where 
    F: OnceDifferentiableFunctionOps<T>,
    W: WeightsInitialization<T>
{
    fn into_config(self) -> LayerConfig<T, F, W> {
        self
    }
}
