use crate::activation::{elu, linear_fn, relu_fn, sigmoid_fn};
use crate::config::Config;
use crate::differentiation::Record;
use crate::function_v2::{He, Softmax, Standard, Xavier};
use crate::mnist::Mnist;
use crate::network::Network;
use crate::utils::{Permutation, PermuteArray};
use clap::Parser;
use ndarray::{ArrayBase, ArrayView1, Axis, Data, Ix1, Zip};
use num_traits::{Float, One, Zero};
use progressing::Baring;
use std::error::Error;
use std::io::Write;
use std::ops::DivAssign;
use tracing::Level;

mod activation;
mod differentiation;
// mod function;
mod config;
mod function_v2;
mod mnist;
mod network;
mod utils;

fn cross_entropy<'a, T>(
    yp: ArrayBase<impl Data<Elem: Into<Record<'a, T>> + Clone>, Ix1>,
    yr: ArrayView1<T>,
) -> Record<'a, T>
where
    T: Copy + Float + DivAssign + One,
{
    let yp = yp.mapv(|x| x.into());
    -Zip::from(&yp)
        .and(&yr)
        .fold(Record::zero(), |acc, yp, &yr| {
            acc + Record::constant(yr)
                * yp.unary(
                    |x| (x + T::epsilon()).ln(),
                    |x| T::one() / (x + T::epsilon()),
                )
        })
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();
    let config = Config::parse();
    let mnist = config.load_mnist_dataset()?;

    let mut network = Network::new(28 * 28)
        .push_hidden_layer(64, (relu_fn(), He))
        .push_output_layer(10, (linear_fn(), Xavier))
        .map_output(Softmax);

    let train_length = mnist.train().length();
    let train_xs = Mnist::features_flattened(mnist.train().features());
    let train_ys = Mnist::targets_unrolled(mnist.train().targets());
    let test_xs = Mnist::features_flattened(mnist.test().features());
    let test_ys = Mnist::targets_unrolled(mnist.test().targets());

    let mut p =
        progressing::mapping::Bar::with_range(0, train_length / config.batch_size * config.epoches)
            .timed();
    p.set_len(45);

    for epoch in 0..config.epoches {
        let permutation = Permutation::random(train_length);
        let train_xs = train_xs.permute_axis(Axis(0), &permutation);
        let train_ys = train_ys.view().permute_axis(Axis(0), &permutation);

        for (i, (xs, ys)) in train_xs
            .axis_chunks_iter(Axis(0), config.batch_size)
            .zip(train_ys.axis_chunks_iter(Axis(0), config.batch_size))
            .enumerate()
        {
            let loss = network.learn(xs, ys, config.learning_rate, cross_entropy);

            p.set(epoch * (train_length / config.batch_size) + i);
            print!("{p:<50}{loss:^10.2}\r");
            std::io::stdout().flush()?;
        }

        let predicted = network.predict_many(test_xs).reversed_axes();
        let error = Zip::from(predicted.rows())
            .and(test_ys.rows())
            .map_collect(|yp, yr| cross_entropy(yp, yr).number)
            .sum();
        println!("{:.3}", predicted.row(0));
        println!("{:.3}", test_ys.row(0));
        println!("{error:<50?}");
    }

    Ok(())
}
