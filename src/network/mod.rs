pub mod config;

use crate::differentiation::{ADTape, FrozenRecord, Indexed, Record, AD};
use crate::function_v2::{
    ArrayFunction, Exp, Linear, OnceDifferentiableFunction, OnceDifferentiableFunctionOps,
    WeightsInitialization,
};
use crate::network::config::{Hidden, IntoLayerConfig, Ready};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, CowArray, Ix1, Ix2, LinalgScalar, Zip};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::RandomExt;
use num_traits::{Float, Zero};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, Deref, DerefMut, Neg};
use serde::de::DeserializeOwned;

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

    #[allow(dead_code)]
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
    #[allow(dead_code)]
    #[inline(always)]
    fn freeze(self) -> Parameters<FrozenRecord<T>> {
        Parameters {
            weights: unsafe { std::mem::transmute(self.weights) },
            biases: unsafe { std::mem::transmute(self.biases) },
        }
    }
}

impl<T: 'static> Layer<T> {
    fn write_to_tape<'a, A>(
        self,
        tape: &'a A::Tape,
    ) -> (OnceDifferentiableFunction<T>, Parameters<A>)
    where
        T: Zero + Clone,
        A: AD<'a, T>,
    {
        (
            self.activation,
            self.parameters.map(|x| A::variable(x, tape)),
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

impl<T, S> Network<T, S> {
    pub fn save_parameters_to(&self, writer: &mut impl std::io::Write) -> Result<(), postcard::Error>
    where
        T: Serialize
    {
        let ps = self.inner.layers.iter().map(|it| &it.parameters).collect::<Vec<_>>();
        postcard::to_io(&ps, writer)?;
        Ok(())
    }
    
    pub fn load_parameters_from(&mut self, reader: &mut impl std::io::Read) -> Result<(), postcard::Error>
    where 
        T: DeserializeOwned
    {
        let mut buffer = [0; 1024 * 1024];
        let (ps, _) = postcard::from_io::<Vec<Parameters<T>>, _>((reader, &mut buffer))?;
        for (layer, p) in self.inner.layers.iter_mut().zip(ps) {
            layer.parameters = p;
        }
        Ok(())
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

    fn copy_to_tape<'a, A>(
        &self,
        tape: &'a A::Tape,
    ) -> (
        TempStorage<OnceDifferentiableFunction<T>>,
        TempStorage<Parameters<A>>,
    )
    where
        A: AD<'a, T>,
        T: Clone + Zero,
    {
        self.inner
            .layers
            .iter()
            .map(|it| it.clone().write_to_tape(tape))
            .unzip()
    }

    // #[instrument(skip_all)]
    fn predict_with_recording<'a, G, A>(
        &self,
        xs: ArrayView2<T>,
        fs: impl IntoIterator<Item = G>,
        ps: &[Parameters<A>],
    ) -> Array2<A>
    where
        A: AD<'a, T> + Exp + LinalgScalar,
        G: OnceDifferentiableFunctionOps<T>,
        T: Clone,
    {
        let mut current = xs.mapv(A::constant);
        for (f, p) in fs.into_iter().zip(ps) {
            let mut a = p.weights.dot(&current);
            Zip::from(a.columns_mut()).for_each(|mut col| col.scaled_add(A::one(), &p.biases));
            let z = a.mapv_into(|x| x.apply_function(&f));
            current = z;
        }
        let mut y_pred = current;
        Zip::from(y_pred.columns_mut()).for_each(|col| self.state.output.call(col));
        y_pred
    }

    // #[instrument(skip_all)]
    fn apply_gradients<'a, A>(
        &mut self,
        total_loss: A,
        learning_rate: T,
        gradients: impl IntoIterator<Item = Parameters<A>>,
    ) where
        T: AddAssign + Float,
        A: AD<'a, T> + Indexed,
    {
        total_loss.with_derivatives(move |ds| {
            for (layer, p) in self.inner.layers.iter_mut().zip(gradients) {
                layer
                    .weights
                    .zip_mut_with(&p.weights, |x, y| *x += learning_rate * ds[y]);
                layer
                    .biases
                    .zip_mut_with(&p.biases, |x, y| *x += learning_rate * ds[y]);
            }
        });
    }

    pub fn learn<'a, A, Tape>(
        &mut self,
        batched_input: ArrayView2<T>,
        batched_target: ArrayView2<T>,
        learning_rate: T,
        loss: impl Fn(ArrayView1<A>, ArrayView1<T>) -> A,
        tape: &'a Tape,
    ) -> (T, Array2<T>)
    where
        T: Float + AddAssign,
        A: AD<'a, T, Tape = Tape> + Exp + Indexed + LinalgScalar + Display,
        Tape: ADTape<T, AD<'a> = A> + 'a,
    {
        debug_assert_eq!(batched_input.nrows(), batched_target.nrows());
        tape.reset();

        let (fs, ps) = self.copy_to_tape(tape);
        let y_pred = self.predict_with_recording(batched_input.reversed_axes(), fs, &ps);

        let total_loss = Zip::from(y_pred.columns())
            .and(batched_target.rows())
            .fold(A::zero(), |acc, yp, yr| acc + loss(yp, yr));

        let factor = T::one() / T::from(batched_target.nrows()).unwrap();
        self.apply_gradients(total_loss, -learning_rate * factor, ps);
        (total_loss.unwrap(), y_pred.mapv(A::unwrap))
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::{linear_fn, relu_fn, sigmoid_fn};
    use crate::differentiation::{Record, WengertList, AD};
    use crate::function_v2::Linear;
    use crate::network::{Hidden, Layer, Network, NetworkData, Parameters, Ready};
    use ndarray::{array, Array1, Array2, ArrayBase, ArrayView1, Data, Dimension, Zip};
    use ndarray_linalg::{aclose, Scalar};
    use ndarray_rand::rand::prelude::StdRng;
    use smallvec::smallvec;

    fn aclose_array<T: Scalar, D: Dimension>(
        actual: ArrayBase<impl Data<Elem = T>, D>,
        expected: ArrayBase<impl Data<Elem = T>, D>,
        atol: T::Real,
    ) {
        Zip::from(&actual)
            .and(&expected)
            .for_each(|a, e| aclose(*a, *e, atol))
    }

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

    fn sse<'a, A>(yp: ArrayView1<A>, yr: ArrayView1<f64>) -> A
    where
        A: AD<'a, f64>,
    {
        Zip::from(&yp).and(&yr).fold(A::zero(), |acc, &yp, &yr| {
            acc + (yp - A::constant(yr)) * (yp - A::constant(yr))
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

        let tape = WengertList::leak(100_000);
        for _ in 0..900 {
            network.learn(xs.view(), ys.view(), 1e-1, sse, tape);
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
            .map_collect(|yp, yr| sse(yp.mapv(Record::constant).view(), yr).number);
        aclose_array(loss_before.view(), array![1.0, 0.5800256583859735], 1e-15);

        let tape = WengertList::leak(100_000);
        let (loss_right_in, _) = network.learn(xs.view(), ys.view(), 0.1, sse, tape);
        aclose(loss_right_in, loss_before.sum(), 1e-15);
        aclose_array(
            network.inner.layers[0].weights.view(),
            array![
                [0.9920037498943847, 0.9920037498943847],
                [0.9920037498943847, 0.9920037498943847]
            ],
            1e-15,
        );
        aclose_array(
            network.inner.layers[0].biases.view(),
            array![-0.03299625010561531, -0.03299625010561531],
            1e-15,
        );
        aclose_array(
            network.inner.layers[1].weights.view(),
            array![[0.882919009282913, 0.882919009282913]],
            1e-15,
        );
        aclose_array(
            network.inner.layers[1].biases.view(),
            array![-0.17615941559557646],
            1e-15,
        );

        let y_pred = network.predict_many(xs.view()).reversed_axes();
        let loss_after = Zip::from(y_pred.rows())
            .and(ys.rows())
            .map_collect(|yp, yr| sse(yp.mapv(Record::constant).view(), yr).number);
        aclose_array(
            loss_after,
            array![0.4791330969810544, 0.13684982266400145],
            1e-15,
        );
    }
}
