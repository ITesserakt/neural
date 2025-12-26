mod config;

use crate::differentiation::{FrozenRecord, Record, WengertList, WengertListPool};
use crate::function_v2::{
    ArrayFunction, Exp, Linear, OnceDifferentiableFunction, OnceDifferentiableFunctionOps,
    WeightsInitialization,
};
use crate::network::config::{Hidden, IntoLayerConfig, Ready};
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, CowArray, Ix1, Ix2, LinalgScalar, Zip};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::RandomExt;
use num_traits::{Float, Zero};
use object_pool::Reusable;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, Deref, DerefMut, Neg};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Parameters<T> {
    weights: Array2<T>,
    biases: Array1<T>,
}

#[derive(Clone)]
struct Layer<T: 'static> {
    activation: OnceDifferentiableFunction<T>,
    size: usize,
    parameters: Parameters<T>,
}

const INLINE_LAYER_BUFFER_SIZE: usize = 3;
type TempStorage<T> = SmallVec<[T; INLINE_LAYER_BUFFER_SIZE]>;

pub(super) struct NetworkData<T: 'static> {
    tapes: WengertListPool<T>,
    layers: SmallVec<[Layer<T>; INLINE_LAYER_BUFFER_SIZE]>,
}

impl<T> NetworkData<T> {
    fn layers(&self) -> &[Layer<T>] {
        self.layers.as_ref()
    }

    fn push_layer(&mut self, value: Layer<T>) {
        self.layers.push(value);
    }

    fn empty() -> Self {
        Self {
            tapes: WengertListPool::new(rayon::max_num_threads()),
            layers: SmallVec::new(),
        }
    }
}

impl<T: 'static> Deref for Layer<T> {
    type Target = Parameters<T>;

    fn deref(&self) -> &Self::Target {
        &self.parameters
    }
}

impl<T: 'static> DerefMut for Layer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.parameters
    }
}

impl<T> Parameters<T> {
    fn map<U>(&self, f: impl Fn(T) -> U) -> Parameters<U>
    where
        T: Clone,
    {
        Parameters {
            weights: self.weights.mapv(&f),
            biases: self.biases.mapv(f),
        }
    }

    fn zeros(output_size: usize, input_size: usize) -> Self
    where
        T: Zero + Clone,
    {
        Self {
            weights: Array2::zeros((output_size, input_size)),
            biases: Array1::zeros(output_size),
        }
    }
}

impl<T> Parameters<Record<'_, T>> {
    #[inline(always)]
    fn freeze(self) -> Parameters<FrozenRecord<T>> {
        Parameters {
            weights: unsafe { std::mem::transmute(self.weights) },
            biases: unsafe { std::mem::transmute(self.biases) },
        }
    }
}

impl<T: 'static> Layer<T> {
    fn write_to_tape(
        self,
        tape: &'_ WengertList<T>,
    ) -> (OnceDifferentiableFunction<T>, Parameters<Record<'_, T>>)
    where
        T: Zero + Clone,
    {
        (
            self.activation,
            self.parameters.map(|x| Record::variable(x, tape)),
        )
    }
}

pub struct Network<T: 'static, State> {
    inner: NetworkData<T>,
    input_size: usize,
    state: State,
}

impl<T> Network<T, Hidden<StdRng>> {
    pub fn new(input_size: usize) -> Self {
        Self {
            inner: NetworkData::empty(),
            input_size,
            state: Hidden {
                rng: StdRng::from_os_rng(),
            },
        }
    }
}

impl<T, R> Network<T, Hidden<R>> {
    #[allow(dead_code)]
    pub fn with_seed(input_size: usize, seed: u64) -> Self
    where
        R: SeedableRng,
        T: Float,
    {
        Self {
            inner: NetworkData::empty(),
            input_size,
            state: Hidden {
                rng: R::seed_from_u64(seed),
            },
        }
    }

    pub fn push_hidden_layer<F, W>(
        mut self,
        size: usize,
        config: impl IntoLayerConfig<T, F, W>,
    ) -> Self
    where
        R: Rng,
        F: OnceDifferentiableFunctionOps<T>,
        W: WeightsInitialization<T>,
    {
        let config = config.into_config();
        let previous_size = self
            .inner
            .layers()
            .last()
            .map_or(self.input_size, |it| it.size);

        let layer = Layer {
            activation: config.activation.into_boxed(),
            size,
            parameters: Parameters {
                weights: Array2::random_using(
                    (size, previous_size),
                    config
                        .weights_initialization
                        .distribution(size, previous_size),
                    &mut self.state.rng,
                ),
                biases: Array1::random_using(
                    size,
                    config
                        .weights_initialization
                        .distribution(size, previous_size),
                    &mut self.state.rng,
                ),
            },
        };
        self.inner.push_layer(layer);

        self
    }

    pub fn push_output_layer<F, W>(
        self,
        size: usize,
        config: impl IntoLayerConfig<T, F, W>,
    ) -> Network<T, Ready<Linear>>
    where
        F: OnceDifferentiableFunctionOps<T>,
        W: WeightsInitialization<T>,
        R: Rng,
    {
        let this = self.push_hidden_layer(size, config);
        Network {
            inner: this.inner,
            input_size: this.input_size,
            state: Ready { output: Linear },
        }
    }
}

impl<T, F> Network<T, Ready<F>>
where
    F: ArrayFunction<Ix1>,
{
    pub fn map_output<G>(self, g: G) -> Network<T, Ready<G>>
    where
        G: ArrayFunction<Ix1>,
    {
        Network {
            inner: self.inner,
            input_size: self.input_size,
            state: Ready { output: g },
        }
    }

    #[allow(dead_code)]
    pub fn predict<'a>(&self, input: impl Into<CowArray<'a, T, Ix1>>) -> Array1<T>
    where
        T: LinalgScalar + Neg<Output = T> + Exp,
    {
        let mut current = input.into();
        debug_assert_eq!(
            self.input_size,
            current.dim(),
            "Network's inputs is {}, but input vector is of size {}",
            self.input_size,
            current.dim()
        );

        for layer in self.inner.layers() {
            let a = layer.weights.dot(&current) + &layer.biases;
            let z = a.mapv_into(|x| layer.activation.function(x));
            current = z.into();
        }

        let mut y_pred = current.to_owned();
        self.state.output.call(y_pred.view_mut());
        y_pred
    }

    pub fn predict_many<'a>(&self, input: impl Into<CowArray<'a, T, Ix2>>) -> Array2<T>
    where
        T: LinalgScalar + Neg<Output = T> + AddAssign + Exp,
    {
        let mut current = input.into().reversed_axes();
        debug_assert_eq!(
            self.input_size,
            current.nrows(),
            "Network's inputs is {}, but input vector is of size {:?}",
            self.input_size,
            current.dim()
        );

        for layer in self.inner.layers() {
            let mut a = layer.weights.dot(&current);
            Zip::from(a.columns_mut()).for_each(|mut a| a += &layer.biases);
            let z = a.mapv_into(|x| layer.activation.function(x));
            current = CowArray::from(z);
        }

        let mut y_pred = current.to_owned();
        Zip::from(y_pred.columns_mut()).for_each(|col| self.state.output.call(col));
        y_pred
    }

    pub fn learn<'a, L>(
        &mut self,
        batched_input: ArrayView2<T>,
        batched_target: ArrayView2<T>,
        learning_rate: T,
        loss: L,
    ) -> T
    where
        T: LinalgScalar + Neg<Output = T> + Float + Sum,
        T: Send + Sync,
        F: Send + Sync,
        L: Fn(Array1<Record<'a, T>>, ArrayView1<T>) -> Record<'a, T>,
        L: Send + Sync,
    {
        debug_assert_eq!(batched_input.nrows(), batched_target.nrows());
        let params = self
            .inner
            .layers
            .iter()
            .cloned()
            .collect::<TempStorage<_>>();
        let (tx, rx) = std::sync::mpsc::channel();

        let compute_gradients =
            |tape: &mut Reusable<&'static WengertList<T>>,
             (x, y): (ArrayView1<T>, ArrayView1<T>)| {
                tape.clear();
                let (fs, ps) = params
                    .iter()
                    .map(|it| it.clone().write_to_tape(tape))
                    .unzip::<_, _, TempStorage<_>, TempStorage<_>>();

                let mut current = x.mapv(Record::constant);
                for (f, p) in fs.iter().zip(&ps) {
                    let a = p.weights.dot(&current) + &p.biases;
                    let z = a.mapv_into(|x| x.unary(|x| f.function(x), |x| f.derivative(x)));
                    current = z;
                }
                let mut y_pred = current;
                self.state.output.call(y_pred.view_mut());
                let loss_value = loss(y_pred, y);

                let ds = loss_value.derivatives();

                (
                    ps.into_iter()
                        .map(|it| it.freeze())
                        .collect::<TempStorage<_>>(),
                    ds,
                    loss_value.number,
                )
            };

        let apply_gradients = || {
            let factor = -learning_rate / T::from(batched_target.nrows()).unwrap();
            for income in rx {
                for (layer, g) in self.inner.layers.iter_mut().zip(income) {
                    let g: Parameters<T> = g;
                    layer.weights.scaled_add(factor, &g.weights);
                    layer.biases.scaled_add(factor, &g.biases);
                }
            }
        };

        let (total_loss, _) = rayon::join(
            || {
                Zip::from(batched_input.rows())
                    .and(batched_target.rows())
                    .into_par_iter()
                    .map_init(|| self.inner.tapes.acquire(), compute_gradients)
                    .map_with(tx, |tx, (ps, ds, loss)| {
                        _ = tx.send(ps.into_iter().map(move |p| Parameters {
                            weights: p.weights.map(|pw| ds[pw]),
                            biases: p.biases.map(|pb| ds[pb]),
                        }));
                        loss
                    })
                    .sum::<T>()
            },
            apply_gradients,
        );

        total_loss
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::{linear_fn, relu_fn, sigmoid_fn};
    use crate::differentiation::{Record, WengertListPool};
    use crate::function_v2::Linear;
    use crate::network::{Hidden, Layer, Network, NetworkData, Parameters, Ready};
    use ndarray::{array, Array1, Array2, ArrayView1, Zip};
    use ndarray_linalg::{aclose, assert_aclose};
    use ndarray_rand::rand::prelude::StdRng;
    use smallvec::smallvec;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_network_prediction() {
        let network = Network::<f64, Hidden<StdRng>>::with_seed(2, 0)
            .push_hidden_layer(4, sigmoid_fn())
            .push_output_layer(1, linear_fn());

        let a = network.predict(array![1.0, 2.0]);
        let b = network.predict(array![3.0, 1.0]);
        let c = network.predict(array![4.0, 3.0]);
        let d = network.predict_many(array![[1.0, 2.0], [3.0, 1.0], [4.0, 3.0]]);

        aclose(d[(0, 0)], a[0], 1e-15);
        aclose(d[(0, 1)], b[0], 1e-15);
        aclose(d[(0, 2)], c[0], 1e-15);
    }

    fn sse<'a>(yp: Array1<Record<'a, f64>>, yr: ArrayView1<f64>) -> Record<'a, f64> {
        Zip::from(&yp)
            .and(&yr)
            .fold(Record::constant(0.0), |acc, &yp, &yr| {
                acc + (yp - Record::constant(yr)) * (yp - Record::constant(yr))
            })
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_network_learning() {
        let mut network = Network::<_, Hidden<StdRng>>::with_seed(2, 0)
            .push_hidden_layer(2, relu_fn())
            .push_output_layer(1, linear_fn());

        let xs = array![
            [1.0, 2.0],
            [0.5, 0.7],
            [0.0, 1.0],
            [2.0, 0.5],
            [-0.15, 0.23]
        ];
        let ys = array![[3.0], [1.2], [1.0], [2.5], [0.08]];

        for _ in 0..900 {
            network.learn(xs.view(), ys.view(), 1e-1, sse);
            println!("{:.3}", network.predict(array![0.1, 0.2]));
        }

        let prediction = network.predict(array![0.1, 0.2]);
        aclose(prediction[0], 0.3, 1e-9);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_network_epoches() {
        let mut network = Network {
            inner: NetworkData {
                tapes: WengertListPool::new(1),
                layers: smallvec![
                    Layer {
                        size: 2,
                        activation: sigmoid_fn(),
                        parameters: Parameters {
                            weights: Array2::ones((2, 2)),
                            biases: Array1::zeros(2),
                        },
                    },
                    Layer {
                        size: 1,
                        activation: linear_fn(),
                        parameters: Parameters {
                            weights: Array2::ones((1, 2)),
                            biases: Array1::zeros(1),
                        },
                    }
                ],
            },
            input_size: 2,
            state: Ready { output: Linear },
        };

        let xs = array![[0.0, 0.0], [1.0, 1.0]];
        let ys = array![[0.0], [1.0]];

        let y_pred = network.predict_many(xs.view()).reversed_axes();
        let loss_before = Zip::from(y_pred.rows())
            .and(ys.rows())
            .map_collect(|yp, yr| sse(yp.mapv(Record::constant), yr).number);
        assert_aclose!(loss_before[0], 1.0, 1e-15);
        assert_aclose!(loss_before[1], 0.5800256583859735, 1e-15);

        let loss_right_in = network.learn(xs.view(), ys.view(), 0.1, sse);
        aclose(loss_right_in, loss_before.sum(), 1e-15);
        assert_eq!(
            network.inner.layers[0].weights,
            array![
                [0.9920037498943847, 0.9920037498943847],
                [0.9920037498943847, 0.9920037498943847]
            ]
        );
        assert_eq!(
            network.inner.layers[0].biases,
            array![-0.03299625010561531, -0.03299625010561531]
        );
        assert_eq!(
            network.inner.layers[1].weights,
            array![[0.882919009282913, 0.882919009282913]]
        );
        assert_eq!(network.inner.layers[1].biases, array![-0.17615941559557646]);

        let y_pred = network.predict_many(xs.view()).reversed_axes();
        let loss_after = Zip::from(y_pred.rows())
            .and(ys.rows())
            .map_collect(|yp, yr| sse(yp.mapv(Record::constant), yr).number);
        assert_aclose!(loss_after[0], 0.4791330969810544, 1e-15);
        assert_aclose!(loss_after[1], 0.13684982266400145, 1e-15);
    }
}
