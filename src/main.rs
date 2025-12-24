use crate::activation::{linear_fn, relu_fn, sigmoid_fn};
use crate::mnist::Mnist;
use crate::network::Network;
use ndarray::{Array2, Axis, Zip};
use ndarray_linalg::Norm;
use ndarray_rand::rand::rngs::StdRng;
use progressing::Baring;
use std::env::args;
use std::io::Write;
use tracing::{info, Level};

mod activation;
mod differentiation;
mod function;
mod mnist;
mod network;

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
        .push_hidden_layer(256, sigmoid_fn())
        .push_output_layer(10, linear_fn());

    let train_length = mnist.train().length();
    let train_xs = Mnist::features_flattened(mnist.train().features());
    let train_ys = Mnist::targets_unrolled(mnist.train().targets());
    let test_xs = Mnist::features_flattened(mnist.test().features());
    let test_ys = Mnist::targets_unrolled(mnist.test().targets());

    let chunks = 10;
    let epoches = 3;
    let mut p = progressing::mapping::Bar::with_range(0, train_length / chunks * epoches).timed();
    p.set_len(45);

    for epoch in 0..epoches {
        for (i, (xs, ys)) in train_xs
            .axis_chunks_iter(Axis(0), chunks)
            .zip(train_ys.axis_chunks_iter(Axis(0), chunks))
            .enumerate()
        {
            network.learn(xs, ys, 5e-6);

            p.set(epoch * (train_length / chunks) + i);
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
            .par_map_collect(|yp, yr| (&yp - &yr).mean().unwrap())
            .mean();
        dbg!(predicted.row(0), test_ys.row(0));
        println!("{error:<50?}");
    }
}
