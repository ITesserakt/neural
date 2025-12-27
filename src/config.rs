use crate::mnist::{Datasets, Mnist, Set};
use clap::Parser;
use ndarray::Dimension;
use numpy::Element;
use pyo3::{PyErr, Python};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, ErrorKind};
use std::path::PathBuf;
use tracing::debug;

#[derive(Debug, Parser, Clone)]
pub struct Config {
    #[arg(long, required = true)]
    dataset_path: PathBuf,
    #[arg(long, default_value = ".cache")]
    cache_path: PathBuf,
    #[arg(long, default_value = "params.dat")]
    pub parameters_path: PathBuf,
    #[arg(short = 'p', long, default_value_t = false)]
    pub load_parameters_from_cache: bool,
    #[arg(short, long, default_value_t = 20)]
    pub epoches: usize,
    #[arg(short, long, default_value_t = 256)]
    pub batch_size: usize,
    #[arg(short, long, default_value_t = 1e-1)]
    pub learning_rate: f32,
}

type SerializationError = postcard::Error;

#[derive(Debug)]
pub enum Error {
    IO(std::io::Error),
    Python(PyErr),
    Serialize(SerializationError),
}

impl Config {
    fn get_cached_dataset_path(&self) -> std::io::Result<PathBuf> {
        match create_dir_all(&self.cache_path) {
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        };
        let dataset_path = self.dataset_path.canonicalize()?;
        let dataset_name = dataset_path
            .file_name()
            .expect("Unknown empty dataset name");
        let path = self.cache_path.join(dataset_name);
        match create_dir_all(&path) {
            Ok(_) => Ok(path),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(path),
            Err(e) => Err(e),
        }
    }

    fn load_dataset_via_python<R>(
        &self,
        callback: impl FnOnce(Datasets) -> Result<R, PyErr>,
    ) -> Result<R, PyErr> {
        Python::initialize();
        Python::attach(|py| Datasets::new(py, &self.dataset_path).and_then(callback))
    }

    fn load_dataset_from_cache<'a, R, DF, DT, T>(
        &self,
        callback: impl FnOnce(HashMap<String, Set<'a, DF, DT, T>>) -> R,
    ) -> std::io::Result<Option<R>>
    where
        DF: Dimension + DeserializeOwned,
        DT: Dimension + DeserializeOwned,
        T: Element + DeserializeOwned,
    {
        let cached_dataset_path = self.get_cached_dataset_path()?;
        if cached_dataset_path.exists() && cached_dataset_path.is_dir() {
            let mut sets = HashMap::new();
            let mut buf = [0; 1024 * 1024];
            for entry in std::fs::read_dir(cached_dataset_path)? {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    continue;
                }

                let file = BufReader::new(File::open(entry.path())?);
                match postcard::from_io::<Set<DF, DT, T>, _>((file, &mut buf)) {
                    Ok((set, _)) => {
                        sets.insert(set.name().to_owned(), set);
                    }
                    Err(e) => debug!("Cannot deserialize {entry:?}: {e}"),
                };
            }

            return Ok(Some(callback(sets)));
        }
        Ok(None)
    }

    pub fn load_mnist_dataset<T>(&self) -> Result<Mnist<'static, T>, Error>
    where
        T: Element + DeserializeOwned + Clone + Serialize,
    {
        let result = self.load_dataset_from_cache(|mut sets| {
            let train = sets.remove("train")?;
            let test = sets.remove("test")?;
            Some(Mnist::from_parts(train, test))
        })?;

        if let Some(Some(mnist)) = result {
            return Ok(mnist);
        }

        let mnist = self
            .load_dataset_via_python(|dataset| dataset.load_as_mnist().map(|it| it.to_owned()))?;
        let cached_dataset_path = self.get_cached_dataset_path()?;
        let mut train_set_file =
            BufWriter::new(File::create(cached_dataset_path.join("train.dat"))?);
        postcard::to_io(mnist.train(), &mut train_set_file)?;
        let mut test_set_file = BufWriter::new(File::create(cached_dataset_path.join("test.dat"))?);
        postcard::to_io(mnist.test(), &mut test_set_file)?;

        Ok(mnist)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::IO(x) => write!(f, "{x}"),
            Error::Python(x) => write!(f, "{x}"),
            Error::Serialize(x) => write!(f, "{x}"),
        }
    }
}

impl std::error::Error for Error {
    fn cause(&self) -> Option<&dyn std::error::Error> {
        match self {
            Error::IO(x) => Some(x),
            Error::Python(x) => Some(x),
            Error::Serialize(x) => Some(x),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::IO(value)
    }
}

impl From<PyErr> for Error {
    fn from(value: PyErr) -> Self {
        Self::Python(value)
    }
}

impl From<SerializationError> for Error {
    fn from(value: SerializationError) -> Self {
        Self::Serialize(value)
    }
}
