use crate::differentiation::Trace;
use crate::function::{Id, OnceDifferentiableFunction};
use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, CowArray, Data, Dimension, Ix1, Ix2, LinalgScalar, ScalarOperand, Zip};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal};
use ndarray_rand::RandomExt;
use num_traits::{ConstOne, ConstZero, Float, FromPrimitive, Zero};
use progressing::Baring;
use std::ops::{AddAssign, Deref, DerefMut, Mul};
use crate::activation::softmax_fn;

pub struct Parameters<T> {
    weights: Array2<T>,
    biases: Array1<T>,
}

pub struct Layer<T: 'static> {
    activation: OnceDifferentiableFunction<'static, (T,)>,
    size: usize,

    parameters: Parameters<T>,
}

pub struct Hidden;
pub struct Ready;

pub struct Network<T: 'static, State, R = StdRng> {
    input_size: usize,
    layers: Vec<Layer<T>>,
    loss: OnceDifferentiableFunction<'static, (Array1<T>, Array1<T>), Id<T>>,
    rng: R,
    _state: State,
}

impl<T> Deref for Layer<T> {
    type Target = Parameters<T>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.parameters
    }
}

impl<T> DerefMut for Layer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.parameters
    }
}

impl<T> Layer<T> {
    #[inline(always)]
    fn activate<D>(&self, x: Array<T, D>) -> Array<T, D>
    where
        T: Clone + Zero,
        D: Dimension,
    {
        x.mapv_into(|it| self.activation.call((it,)))
    }

    #[inline(always)]
    fn activate_with_derivative<D>(
        &self,
        x: ArrayBase<impl Data<Elem = T>, D>,
    ) -> Array<Trace<T>, D>
    where
        T: Clone + ConstOne,
        D: Dimension,
    {
        x.mapv(|it| self.activation.derivative(it))
    }
}

impl<T> Network<T, Hidden> {
    pub fn new(input_size: usize) -> Self
    where
        T: FromPrimitive,
        T: Float,
        T: Send + Sync,
    {
        Self {
            input_size,
            layers: vec![],
            rng: StdRng::from_os_rng(),
            loss: Self::DEFAULT_LOSS_FUNCTION,
            _state: Hidden,
        }
    }
}

impl<T, S, R> Network<T, S, R>
where
    T: Float,
{
    const DEFAULT_LOSS_FUNCTION: OnceDifferentiableFunction<'static, (Array1<T>, Array1<T>), Id<T>> =
        OnceDifferentiableFunction::from_ref(
            &|(x, y): (Array1<T>, Array1<T>)| (&x - &y).powi(2).sum(),
            &|_: (Trace<Array1<T>>, Trace<Array1<T>>)| todo!(),
        );
}

impl<T, R> Network<T, Hidden, R> {
    pub fn with_seed(input_size: usize, seed: u64) -> Self
    where
        R: SeedableRng,
        T: Float,
    {
        Self {
            input_size,
            layers: vec![],
            rng: R::seed_from_u64(seed),
            loss: Self::DEFAULT_LOSS_FUNCTION,
            _state: Hidden,
        }
    }
}

impl<T, R> Network<T, Hidden, R>
where
    R: Rng,
{
    pub fn push_hidden_layer(
        mut self,
        size: usize,
        activation: OnceDifferentiableFunction<'static, (T,)>,
    ) -> Self
    where
        StandardNormal: Distribution<T>,
        T: Float,
    {
        let normal = Normal::new(T::zero(), T::one()).unwrap();
        let previous_size = self.layers.last().map_or(self.input_size, |it| it.size);

        self.layers.push(Layer {
            activation,
            size,
            parameters: Parameters {
                weights: Array2::random_using(
                    (size, previous_size),
                    normal.map(|it| it * (T::from(2.0 / (size + previous_size) as f64).unwrap().sqrt())),
                    &mut self.rng,
                ),
                biases: Array1::random_using(
                    size,
                    normal.map(|it| it * (T::from(2.0 / size as f64).unwrap().sqrt())),
                    &mut self.rng,
                ),
            },
        });
        self
    }

    pub fn push_output_layer(
        self,
        size: usize,
        activation: OnceDifferentiableFunction<'static, (T,)>,
    ) -> Network<T, Ready, R>
    where
        T: Float,
        StandardNormal: Distribution<T>,
    {
        let this = self.push_hidden_layer(size, activation);
        Network {
            layers: this.layers,
            input_size: this.input_size,
            loss: Self::DEFAULT_LOSS_FUNCTION,
            rng: this.rng,
            _state: Ready,
        }
    }
}

impl<T, R> Network<T, Ready, R> {
    fn empty_parameters(&self) -> impl Iterator<Item = Parameters<T>>
    where
        T: Zero + Clone,
    {
        self.layers.iter().map(|it| Parameters {
            weights: Array2::zeros(it.weights.dim()),
            biases: Array1::zeros(it.biases.dim()),
        })
    }

    pub fn predict<'a>(&self, input: impl Into<CowArray<'a, T, Ix1>>) -> Array1<T>
    where
        T: LinalgScalar,
    {
        let mut current = input.into();
        debug_assert_eq!(
            self.input_size,
            current.dim(),
            "Network's inputs is {}, but input vector is of size {}",
            self.input_size,
            current.dim()
        );

        for layer in self.layers.iter() {
            let a = layer.weights.dot(&current) + &layer.biases;
            let z = layer.activate(a);
            current = CowArray::from(z);
        }
        current.to_owned()
    }

    pub fn predict_many<'a>(&self, inputs: impl Into<CowArray<'a, T, Ix2>>) -> Array2<T>
    where
        T: LinalgScalar + AddAssign,
    {
        let mut current = inputs.into().reversed_axes();
        debug_assert_eq!(
            self.input_size,
            current.nrows(),
            "Network's inputs is {}, but input vector is of size {:?}",
            self.input_size,
            current.dim()
        );

        for layer in self.layers.iter() {
            let mut a = layer.weights.dot(&current);
            Zip::from(a.columns_mut()).for_each(|mut a| a += &layer.biases);
            let z = layer.activate(a);
            current = CowArray::from(z);
        }

        current.to_owned()
    }

    #[inline(always)]
    fn compute_partial_derivatives<'a, 'b>(
        &'a self,
        x: ArrayView1<'b, T>,
        y: ArrayView1<'b, T>,
    ) -> impl Iterator<Item = (Array1<T>, Array1<T>)> + use<'a, T, R>
    where
        T: Float + ConstOne + ConstZero + ScalarOperand,
    {
        let mut current = CowArray::from(x);
        let (zss, mut ass) = self
            .layers
            .iter()
            .map(|layer| {
                let zs = layer.weights.dot(&current) + &layer.biases;
                let ass = layer.activate(zs.to_owned());
                current = ass.to_owned().into();
                (zs, ass)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let a = ass.pop().unwrap();
        let mut loss_grad = Zip::from(&a).and(&y).map_collect(|yp, yr| *yp - *yr);

        let deltas = self
            .layers
            .iter()
            .zip(zss)
            .rev()
            .map(|(layer, zs)| {
                let delta = zs
                    .mapv_into(|z| layer.activation.derivative(z).derivative)
                    .mul(&loss_grad);

                loss_grad = layer.weights.t().dot(&delta);
                delta
            })
            .collect::<Vec<_>>();

        deltas
            .into_iter()
            .zip(std::iter::once(x.to_owned()).chain(ass).rev())
    }

    pub fn learn(
        &mut self,
        batched_input: ArrayView2<T>,
        batched_target: ArrayView2<T>,
        learning_rate: T,
    ) where
        T: Float + AddAssign + ConstOne + ConstZero + ScalarOperand + Send + Sync,
        R: Send + Sync,
    {
        assert_eq!(batched_input.nrows(), batched_target.nrows());

        let init = || self.empty_parameters().collect::<Vec<_>>();
        let combine = |a: Vec<Parameters<T>>, b: Vec<Parameters<T>>| -> Vec<_> {
            a.into_iter()
                .zip(b)
                .map(|(mut a, b)| {
                    a.weights += &b.weights;
                    a.biases += &b.biases;
                    a
                })
                .collect()
        };
        let gradients = Zip::from(batched_input.rows())
            .and(batched_target.rows())
            .par_fold(
                init,
                |mut acc, x, y| {
                    self.compute_partial_derivatives(x, y)
                        .zip(acc.iter_mut().rev())
                        .for_each(|((d, a), acc)| {
                            let a_len = a.len();
                            let at = a.into_shape_with_order((1, a_len)).unwrap();
                            let d_len = d.len();
                            let dt = d.into_shape_with_order((d_len, 1)).unwrap();
                            acc.weights += &dt.dot(&at);
                            acc.biases += &dt.into_flat();
                        });
                    acc
                },
                combine,
            );

        let scale = learning_rate / T::from(batched_input.nrows()).unwrap();
        self.layers
            .iter_mut()
            .zip(gradients)
            .for_each(|(layer, p)| {
                layer.weights.scaled_add(-scale, &p.weights);
                layer.biases.scaled_add(-scale, &p.biases);
            });
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::{linear_fn, sigmoid_fn};
    use crate::network::Network;
    use ndarray::array;
    use ndarray_linalg::aclose;
    use ndarray_rand::rand::rngs::StdRng;

    #[test]
    fn test_network_prediction() {
        let network = Network::<f64, _, StdRng>::with_seed(2, 0)
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
}
