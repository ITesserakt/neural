use crate::activation::{linear_fn, sigmoid_fn};
use crate::config::Config;
use crate::differentiation::{Adr, Record, WengertListPool, AD};
use crate::function_v2::{ArrayFunction, OnceDifferentiableFunction, Softmax};
use crate::mnist::Mnist;
use crate::network::config::Ready;
use crate::network::Network;
use crate::utils::{Permutation, PermuteArray};
use clap::Parser;
use indicatif::style::TemplateError;
use indicatif::ProgressStyle;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis, Ix1, Zip};
use ndarray_linalg::Scalar;
use num_traits::{Float, One, Zero};
use numpy::Element;
use std::error::Error;
use std::fmt::Write;
use std::ops::{DivAssign, Neg};
use tracing::{instrument, Level, Span};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

mod activation;
mod config;
mod differentiation;
mod function_v2;
mod mnist;
mod network;
mod utils;

struct Env<'a, T: 'static, F> {
    network: Network<T, Ready<F>>,
    train_xs: ArrayView2<'a, T>,
    train_ys: Array2<T>,
    test_xs: ArrayView2<'a, T>,
    test_ys: Array2<T>,
    config: Config,
    tapes: WengertListPool<T>,
}

impl<'a, T: Element + Scalar, F> Env<'a, T, F> {
    pub fn new(network: Network<T, Ready<F>>, mnist: &'a Mnist<T>, config: Config) -> Self {
        Self {
            network,
            train_xs: Mnist::features_flattened(mnist.train().features()),
            train_ys: Mnist::targets_unrolled(mnist.train().targets()),
            test_xs: Mnist::features_flattened(mnist.test().features()),
            test_ys: Mnist::targets_unrolled(mnist.test().targets()),
            config,
            tapes: WengertListPool::new(1),
        }
    }
}

impl<F: ArrayFunction<Ix1> + Send + Sync> Env<'_, f32, F> {
    fn compute_loss(&self, xs: ArrayView2<f32>, ys: ArrayView2<f32>) -> f32
    where
        F: ArrayFunction<Ix1> + Send + Sync,
    {
        let y_pred = self.network.predict_many(xs).reversed_axes();
        Zip::from(y_pred.rows())
            .and(ys.rows())
            .fold(f32::zero(), |acc, yp, yr| {
                acc + cross_entropy(yp.mapv(Record::constant).view(), yr).number
            })
    }

    #[instrument(skip_all)]
    fn batch_iterations(&mut self, xs: ArrayView2<f32>, ys: ArrayView2<f32>)
    where
        F: ArrayFunction<Ix1> + Send + Sync,
    {
        let mut message = String::new();
        Span::current().pb_set_length(xs.nrows() as u64 / self.config.batch_size as u64);
        for (xs, ys) in xs
            .axis_chunks_iter(Axis(0), self.config.batch_size)
            .zip(ys.axis_chunks_iter(Axis(0), self.config.batch_size))
        {
            let tape = self.tapes.acquire();
            let loss = self.network.learn::<Record<_>>(
                xs,
                ys,
                self.config.learning_rate,
                cross_entropy,
                *tape,
            );

            message.clear();
            write!(&mut message, "Loss = {loss:.3}").unwrap();
            Span::current().pb_set_message(&message);
            Span::current().pb_inc(1);
        }
    }

    #[instrument(skip_all)]
    fn epoch_iterations(&mut self)
    where
        F: ArrayFunction<Ix1> + Send + Sync,
    {
        Span::current().pb_set_length(self.config.epoches as u64);
        for _ in 0..self.config.epoches {
            let permutation = Permutation::random(self.train_xs.nrows());
            let xs = self.train_xs.permute_axis(Axis(0), &permutation);
            let ys = self.train_ys.view().permute_axis(Axis(0), &permutation);

            self.batch_iterations(xs.view(), ys.view());

            let loss = self.compute_loss(self.test_xs.view(), self.test_ys.view());

            Span::current().pb_set_message(&format!("Loss = {loss:.3}"));
            Span::current().pb_inc(1);
        }
    }
}

fn cross_entropy<'a, T, A>(yp: ArrayView1<A>, yr: ArrayView1<T>) -> A
where
    T: Copy + Float + DivAssign + One + 'static,
    A: AD<T> + Neg<Output = A>,
{
    let ln_fn = OnceDifferentiableFunction::from_static(
        &|x: T| (x + T::epsilon()).ln(),
        &|x: T| T::one() / (x + T::epsilon()),
        &|x| (x + Adr::epsilon()).ln(),
    );

    -Zip::from(&yp).and(&yr).fold(A::zero(), |acc, yp, &yr| {
        acc + A::constant(yr) * yp.apply_function(&ln_fn)
    })
}

fn init_tracing() -> Result<(), TemplateError> {
    let indicatif_layer = IndicatifLayer::new().with_progress_style(
        ProgressStyle::with_template(
            "{span_child_prefix} {span_name} [{elapsed_precise}] [{bar:.cyan/blue}] {{{msg}}} ({pos}/{len} | {eta})",
        )?
        .progress_chars("#>-"),
    );

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(indicatif_layer)
        .with(
            EnvFilter::builder()
                .with_default_directive(Level::DEBUG.into())
                .from_env_lossy(),
        )
        .init();

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    init_tracing()?;
    let config = Config::parse();
    let mnist = config.load_mnist_dataset()?;

    let network = Network::new(28 * 28)
        .push_hidden_layer(32, sigmoid_fn())
        .push_output_layer(10, linear_fn())
        .map_output(Softmax);

    let mut env = Env::new(network, &mnist, config.clone());
    env.epoch_iterations();

    Ok(())
}
