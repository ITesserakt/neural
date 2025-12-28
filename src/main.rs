use crate::activation::{gaussian_fn, leaky_relu, linear_fn, relu_fn, silu, softmax};
use crate::config::Config;
use crate::function_v2::{ArrayFunction, He, Softmax};
use crate::mnist::Mnist;
use crate::network::config::Ready;
use crate::network::Network;
use crate::utils::{Permutation, PermuteArray};
use auto_differentiation::record::{Record, WengertList};
use auto_differentiation::{Exp, AD};
use clap::Parser;
use indicatif::ProgressStyle;
use ndarray::{s, Array2, ArrayView1, ArrayView2, Axis, Ix1, Zip};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use numpy::Element;
use std::error::Error;
use std::fmt::{Display, Write};
use std::fs::File;
use std::io::{stdin, stdout, BufReader, BufWriter};
use std::ops::{AddAssign, DivAssign, Neg};
use std::str::FromStr;
use serde::Serialize;
use tracing::level_filters::LevelFilter;
use tracing::{error, info, info_span, instrument, trace, warn, Span, Subscriber};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::fmt::format;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

mod activation;
mod config;
mod function_v2;
mod mnist;
mod network;
mod utils;

struct Env<'a, T, F>
where
    T: FromStr<Err: Into<Box<dyn Error + Send + Sync + 'static>>>,
    T: Copy + 'static + Send + Sync,
{
    network: Network<T, Ready<F>>,
    train_xs: ArrayView2<'a, T>,
    train_ys: Array2<T>,
    test_xs: ArrayView2<'a, T>,
    test_ys: Array2<T>,
    config: Config<T>,
}

impl<'a, T, F> Env<'a, T, F>
where
    T: FromStr<Err: Into<Box<dyn Error + Send + Sync + 'static>>>,
    T: Element + Copy + Send + Sync + 'static + ToPrimitive + One + Zero,
{
    pub fn new(network: Network<T, Ready<F>>, mnist: &'a Mnist<T>, config: Config<T>) -> Self {
        Self {
            network,
            train_xs: Mnist::features_flattened(mnist.train().features()),
            train_ys: Mnist::targets_unrolled(mnist.train().targets()),
            test_xs: Mnist::features_flattened(mnist.test().features()),
            test_ys: Mnist::targets_unrolled(mnist.test().targets()),
            config,
        }
    }
}

impl<F: ArrayFunction<Ix1> + Send + Sync, T> Env<'_, T, F>
where
    T: FromStr<Err: Into<Box<dyn Error + Send + Sync + 'static>>>,
    T: Element
        + Exp
        + Float
        + Copy
        + Send
        + Sync
        + 'static
        + AddAssign
        + DivAssign
        + Display
        + FromPrimitive
        + Serialize,
{
    fn compute_loss(&self, xs: ArrayView2<T>, ys: ArrayView2<T>) -> (T, Array2<T>)
    where
        F: ArrayFunction<Ix1> + Send + Sync,
    {
        let y_pred = self.network.predict_many(xs).reversed_axes();
        let loss = Zip::from(y_pred.rows())
            .and(ys.rows())
            .fold(T::zero(), |acc, yp, yr| {
                acc + cross_entropy(yp.mapv(Record::constant).view(), yr).number
            });
        (loss, y_pred)
    }

    #[instrument(skip_all)]
    fn batch_iterations(
        &mut self,
        xs: ArrayView2<T>,
        ys: ArrayView2<T>,
        tape: &'static WengertList<T>,
    ) where
        F: ArrayFunction<Ix1> + Send + Sync,
    {
        let mut message = String::new();
        Span::current().pb_set_length(xs.nrows() as u64 / self.config.batch_size as u64);

        for (xs, ys) in xs
            .axis_chunks_iter(Axis(0), self.config.batch_size)
            .zip(ys.axis_chunks_iter(Axis(0), self.config.batch_size))
        {
            let (loss, _) =
                self.network
                    .learn(xs, ys, self.config.learning_rate, cross_entropy, tape);

            message.clear();
            trace!(target: "output", "{loss}");
            write!(&mut message, "Loss = {loss:.3}").unwrap();
            Span::current().pb_set_message(&message);
            Span::current().pb_inc(1);
        }
    }

    #[instrument(skip_all)]
    fn epoch_iterations(&mut self, tape: &'static WengertList<T>)
    where
        F: ArrayFunction<Ix1> + Send + Sync,
    {
        Span::current().pb_set_length(self.config.epoches as u64);
        for _ in 0..self.config.epoches {
            let permutation = Permutation::random(self.train_xs.nrows());
            let xs = self.train_xs.permute_axis(Axis(0), &permutation);
            let ys = self.train_ys.view().permute_axis(Axis(0), &permutation);

            self.batch_iterations(xs.view(), ys.view(), tape);

            let (loss, y_pred) = self.compute_loss(self.test_xs.view(), self.test_ys.view());
            for item in y_pred {
                trace!(target: "predictions", "{item}");
            }

            Span::current().pb_set_message(&format!("Loss = {loss:.3}"));
            Span::current().pb_inc(1);
        }
    }

    fn answer_prompt(&self) -> std::io::Result<bool> {
        let mut prompt = String::new();
        print!("> ");
        std::io::Write::flush(&mut stdout())?;
        if stdin().read_line(&mut prompt)? == 0 {
            return Ok(false);
        }

        match prompt.trim() {
            x if x.starts_with("test all") => {
                let y_pred = self.network.predict_many(self.test_xs);
                let loss = Zip::from(y_pred.columns())
                    .and(self.test_ys.rows())
                    .map_collect(|yp, yr| {
                        cross_entropy(yp.mapv(Record::constant).view(), yr).number
                    });

                info!("Mean loss  = {}", loss.mean().unwrap());
                info!("Loss std   = {}", loss.std(T::zero()));
                info!("Total loss = {}", loss.sum());
            }
            x if x.starts_with("test") => {
                let index_str = x.strip_prefix("test").unwrap().trim();
                let Ok(index) = index_str.parse::<usize>() else {
                    warn!("Cannot parse: {index_str}");
                    return Ok(true);
                };

                let y_real = self.test_ys.slice(s![index, ..]);
                let x = self.test_xs.slice(s![index, ..]);
                let y_pred = self.network.predict(x);
                let loss = cross_entropy(y_pred.mapv(Record::constant).view(), y_real).number;

                info!("Expected target probabilities:  {y_real:.2}");
                info!("Predicted target probabilities: {y_pred:.2}");
                info!("Loss: {}", loss)
            }
            "save" => {
                info!("Saving network parameters");
                let mut file = BufWriter::new(File::create(&self.config.parameters_path)?);
                if let Err(e) = self.network.save_parameters_to(&mut file) {
                    error!("Cannot serialize parameters: {e}")
                }
            }
            "quit" => return Ok(false),
            _ => {}
        };

        Ok(true)
    }
}

fn cross_entropy<'a, T, A>(yp: ArrayView1<A>, yr: ArrayView1<T>) -> A
where
    T: Float + DivAssign + 'static,
    A: AD<'a, T> + Neg<Output = A>,
{
    -Zip::from(&yp).and(&yr).fold(A::zero(), |acc, yp, &yr| {
        acc + A::constant(yr) * yp.apply_function(|x| x.ln(), |x| T::one() / x)
    })
}

fn init_tracing() -> Result<(), Box<dyn Error>> {
    let indicatif_layer = IndicatifLayer::new().with_progress_style(
        ProgressStyle::with_template(
            "{span_child_prefix} {span_name} [{elapsed_precise}] [{bar:.cyan/blue}] {{{msg}}} ({pos}/{len} | {eta})",
        )?
        .progress_chars("#>-"),
    );

    let output = File::create("output.csv")?;
    let predictions = File::create("predictions.csv")?;
    let ys_test = File::create("ys_test.csv")?;

    fn create_file_output_layer<S: Subscriber>(file: File, name: &'static str) -> impl Layer<S>
    where
        for<'span> S: LookupSpan<'span>,
    {
        tracing_subscriber::fmt::layer()
            .with_writer(file)
            .event_format(format().compact())
            .with_level(false)
            .with_target(false)
            .with_ansi(true)
            .without_time()
            .with_filter(LevelFilter::TRACE)
            .with_filter(filter_fn(move |m| m.target() == name))
    }

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(indicatif_layer.get_stderr_writer())
                .with_filter(LevelFilter::DEBUG),
        )
        .with(indicatif_layer)
        .with(create_file_output_layer(predictions, "predictions"))
        .with(create_file_output_layer(output, "output"))
        .with(create_file_output_layer(ys_test, "ys_test"))
        .init();

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    init_tracing()?;
    let config = Config::<f32>::parse();
    let mnist = config.load_mnist_dataset()?;

    {
        for item in mnist.test().targets() {
            trace!(target: "ys_test", "{item}");
        }
    }

    let mut network = Network::new(28 * 28)
        .push_hidden_layer(32, (leaky_relu(), He))
        .push_output_layer(10, linear_fn())
        .map_output(Softmax);

    if config.load_parameters_from_cache {
        let mut file = BufReader::new(File::open(&config.parameters_path)?);
        network.load_parameters_from(&mut file)?;
    }

    let mut env = Env::new(network, &mnist, config.clone());
    {
        let _output_collector = info_span!("output").entered();
        trace!(target: "output", "loss");
        let _predictions_collector = info_span!("predictions").entered();
        if !config.load_parameters_from_cache {
            let tape = WengertList::leak(1 << 28);
            env.epoch_iterations(tape);
        }
    }

    while env.answer_prompt()? {}

    Ok(())
}
