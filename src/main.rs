use crate::activation::{linear_fn, relu_fn, sigmoid_fn};
use crate::mnist::Mnist;
use crate::network::Network;
use ndarray::{Array1, ArrayView1, Axis, Slice, Zip};
use progressing::Baring;
use std::env::args;
use std::io::Write;
use num_traits::Float;
use tracing::{info, Level};
use crate::utils::{Permutation, PermuteArray};

mod activation;
mod differentiation;
mod function;
mod mnist;
mod network;
mod utils;

enum Commands {

}

fn cross_entropy<T: Float + 'static>(yp: ArrayView1<T>, yr: ArrayView1<T>) -> T {
    Zip::from(&yp).and(&yr).fold(T::zero(), |acc, yp, yr| {
        acc - *yr * (*yp + T::epsilon()).ln()
    })
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();
    let mut args = args();
    args.next();
    let dataset_path = args
        .next()
        .expect("Dataset path should be the first argument");

    let mnist = match Mnist::load(dataset_path) {
        Ok(x) => x,
        Err(e) => {
            println!("Python error: {e}");
            return;
        }
    };
    info!(
        "Successfully extracted dataset with trains = {}, tests = {}",
        mnist.train().length(),
        mnist.test().length()
    );

    let mut network = Network::new(28 * 28)
        .push_hidden_layer(32, sigmoid_fn())
        .push_output_layer(10, linear_fn());

    let train_length = mnist.train().length();
    let train_xs = Mnist::features_flattened(mnist.train().features());
    let train_ys = Mnist::targets_unrolled(mnist.train().targets());
    let test_xs = Mnist::features_flattened(mnist.test().features());
    let test_ys = Mnist::targets_unrolled(mnist.test().targets());

    let batch_size = 1000;
    let epoches = 20;
    let mut p = progressing::mapping::Bar::with_range(0, train_length / batch_size * epoches).timed();
    p.set_len(45);

    for epoch in 0..epoches {
        let permutation = Permutation::random(train_length);
        let train_xs = train_xs.permute_axis(Axis(0), &permutation);
        let train_ys = train_ys.view().permute_axis(Axis(0), &permutation);

        for (i, (xs, ys)) in train_xs
            .axis_chunks_iter(Axis(0), batch_size)
            .zip(train_ys.axis_chunks_iter(Axis(0), batch_size))
            .enumerate()
        {
            network.learn(xs, ys, 1e-5);

            p.set(epoch * (train_length / batch_size) + i);
            print!("{p:<50}\r");
            std::io::stdout().flush().unwrap();
        }

        let mut predicted = network.predict_many(test_xs).reversed_axes();
        Zip::from(predicted.rows_mut()).par_for_each(|mut row| {
            row.mapv_inplace(f64::exp);
            let sum = row.sum();
            row /= sum;
        });
        let error = Zip::from(predicted.rows())
            .and(test_ys.rows())
            .par_map_collect(cross_entropy)
            .mean();
        println!("{:.3}", predicted.row(0));
        println!("{:.3}", test_ys.row(0));
        println!("{error:<50?}");
    }
}
