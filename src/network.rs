use crate::differentiation::{Record, WengertList};
use crate::function_v2::{ArrayFunction, Exp, Linear, OnceDifferentiableFunction};
use crate::network::hacks::NetworkData;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, CowArray, Ix1, Ix2, LinalgScalar, Zip};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal};
use ndarray_rand::RandomExt;
use num_traits::{Float, Zero};
use smallvec::SmallVec;
use std::fmt::Debug;
use std::ops::{AddAssign, Deref, DerefMut, Neg, SubAssign};

#[derive(Debug, Clone)]
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

pub struct Hidden<R> {
    rng: R,
}
pub struct Ready<F> {
    output: F,
}

mod hacks {
    use crate::differentiation::WengertList;
    use crate::network::{Layer, INLINE_LAYER_BUFFER_SIZE};
    use smallvec::SmallVec;

    pub(super) struct NetworkData<T: 'static> {
        pub(super) tape: &'static WengertList<T>,
        layers: SmallVec<[Layer<T>; INLINE_LAYER_BUFFER_SIZE]>,
    }

    impl<T> NetworkData<T> {
        pub(super) fn layers(&self) -> &[Layer<T>] {
            self.layers.as_ref()
        }

        pub(super) fn layers_mut(&mut self) -> &mut [Layer<T>] {
            self.layers.as_mut()
        }

        pub(super) fn push_layer(&mut self, value: Layer<T>) {
            self.layers.push(value);
        }

        pub(super) fn empty() -> Self {
            Self {
                tape: WengertList::leak(),
                layers: SmallVec::new(),
            }
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

    pub fn push_hidden_layer(
        mut self,
        size: usize,
        activation: OnceDifferentiableFunction<T>,
    ) -> Self
    where
        R: Rng,
        T: Float,
        StandardNormal: Distribution<T>,
    {
        let normal = Normal::new(T::zero(), T::one()).unwrap();
        let previous_size = self
            .inner
            .layers()
            .last()
            .map_or(self.input_size, |it| it.size);

        let w = T::from(2.0 / (size + previous_size) as f64).unwrap().sqrt();
        let weights = Array2::random_using(
            (size, previous_size),
            normal.map(|x| x * w),
            &mut self.state.rng,
        );
        let b = T::from(2.0 / size as f64).unwrap().sqrt();
        let biases = Array1::random_using(size, normal.map(|x| x * b), &mut self.state.rng);

        let layer = Layer {
            activation,
            size,
            parameters: Parameters { weights, biases },
        };
        self.inner.push_layer(layer);

        self
    }

    pub fn push_output_layer(
        self,
        size: usize,
        activation: OnceDifferentiableFunction<T>,
    ) -> Network<T, Ready<Linear>>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        R: Rng,
    {
        let this = self.push_hidden_layer(size, activation);
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
            let z = a.mapv_into(|x| layer.activation.call(x));
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
            let z = a.mapv_into(|x| layer.activation.call(x));
            current = CowArray::from(z);
        }

        let mut y_pred = current.to_owned();
        Zip::from(y_pred.columns_mut()).for_each(|col| self.state.output.call(col));
        y_pred
    }

    pub fn learn<'a>(
        &mut self,
        batched_input: ArrayView2<T>,
        batched_target: ArrayView2<T>,
        learning_rate: T,
        loss: impl Fn(Array1<Record<'a, T>>, ArrayView1<T>) -> Record<'a, T>,
    ) -> Array1<T>
    where
        T: LinalgScalar + Neg<Output = T> + SubAssign + Float,
        T: Debug,
    {
        assert_eq!(batched_input.nrows(), batched_target.nrows());

        Zip::from(batched_input.rows())
            .and(batched_target.rows())
            .map_collect(|x, y| {
                let (fs, ps) = self
                    .inner
                    .layers()
                    .iter()
                    .map(|it| it.clone().write_to_tape(self.inner.tape))
                    .unzip::<_, _, SmallVec<[_; INLINE_LAYER_BUFFER_SIZE]>, SmallVec<[_; INLINE_LAYER_BUFFER_SIZE]>>();

                let mut current = x.mapv(Record::constant);
                for (f, p) in fs.iter().zip(&ps) {
                    let a = p.weights.dot(&current) + &p.biases;
                    let z = a.mapv_into(|x| x.unary(|x| f.call(x), |x| f.derivative(x)));
                    current = z;
                }
                let mut y_pred = current;
                self.state.output.call(y_pred.view_mut());
                let loss_value = loss(y_pred, y);

                let ds = loss_value.derivatives();
                for (layer, p) in self.inner.layers_mut().into_iter().zip(ps) {
                    Zip::from(&mut layer.weights)
                        .and(&p.weights)
                        .for_each(|w, pw| *w -= learning_rate * ds[pw]);

                    Zip::from(&mut layer.biases)
                        .and(&p.biases)
                        .for_each(|b, pb| *b -= learning_rate * ds[pb]);
                }
                self.inner.tape.clear();

                loss_value.number
            })
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::{linear_fn, relu_fn, sigmoid_fn};
    use crate::differentiation::Record;
    use crate::network::{Hidden, Network};
    use ndarray::{array, Array1, ArrayView1, Zip};
    use ndarray_linalg::aclose;
    use ndarray_rand::rand::prelude::StdRng;

    #[test]
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

    #[test]
    fn test_network_learning() {
        fn sse<'a>(yp: Array1<Record<'a, f64>>, yr: ArrayView1<f64>) -> Record<'a, f64> {
            Zip::from(&yp)
                .and(&yr)
                .fold(Record::constant(0.0), |acc, &yp, &yr| {
                    acc + Record::constant(0.5)
                        * (yp - Record::constant(yr)).unary(|x| x.powi(2), |x| 2.0 * x)
                })
        }

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

        for _ in 0..300 {
            println!(
                "{:.3}\t\t{:.3}",
                network.learn(xs.view(), ys.view(), 1e-1, sse),
                network.predict(array![0.1, 0.2])
            );
        }

        let prediction = network.predict(array![0.1, 0.2]);
        aclose(prediction[0], 0.3, 1e-9);
    }
}
