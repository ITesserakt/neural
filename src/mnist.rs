use ndarray::{
    Array, Array2, ArrayView, ArrayView1, ArrayView2, ArrayView3, Dimension, Ix1, Ix3, Zip,
};
use num_traits::{One, ToPrimitive, Zero};
use numpy::{dtype, Element, PyArrayMethods, PyReadonlyArray};
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyDict;
use pyo3::{Bound, PyAny, PyErr, Python};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::debug;

#[derive(Debug, Clone)]
enum ArrayStorage<'py, T: Element, D: Dimension> {
    Python(PyReadonlyArray<'py, T, D>),
    Rust(Array<T, D>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(from = "SerializableSet<DF, DT, T>")]
#[serde(into = "SerializableSet<DF, DT, T>")]
#[serde(bound(serialize = "T: Clone + Serialize, DF: Serialize, DT: Serialize"))]
pub struct Set<'py, DF: Dimension, DT: Dimension, T: Element> {
    name: String,
    features: ArrayStorage<'py, T, DF>,
    targets: ArrayStorage<'py, T, DT>,
    num_rows: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SerializableSet<DF: Dimension, DT: Dimension, T: Element> {
    name: String,
    features: Array<T, DF>,
    targets: Array<T, DT>,
    num_rows: usize,
}

pub type MnistSet<'a, T> = Set<'a, Ix3, Ix1, T>;

#[derive(Debug)]
pub struct Mnist<'a, T: Element> {
    train_set: MnistSet<'a, T>,
    test_set: MnistSet<'a, T>,
}

struct Dataset<'py> {
    features: Bound<'py, PyAny>,
    num_rows: usize,
}

impl<DF: Dimension, DT: Dimension, T: Element> From<SerializableSet<DF, DT, T>>
    for Set<'static, DF, DT, T>
{
    fn from(value: SerializableSet<DF, DT, T>) -> Self {
        Self {
            name: value.name,
            features: ArrayStorage::Rust(value.features),
            targets: ArrayStorage::Rust(value.targets),
            num_rows: value.num_rows,
        }
    }
}

impl<'a, DF: Dimension, DT: Dimension, T: Element> From<Set<'a, DF, DT, T>>
    for SerializableSet<DF, DT, T>
where
    T: Clone,
{
    fn from(value: Set<'a, DF, DT, T>) -> Self {
        Self {
            name: value.name,
            features: value.features.view().to_owned(),
            targets: value.targets.view().to_owned(),
            num_rows: value.num_rows,
        }
    }
}

impl<'a, T: Element> Mnist<'a, T> {
    pub fn train(&self) -> &MnistSet<'a, T> {
        &self.train_set
    }

    pub fn test(&self) -> &MnistSet<'a, T> {
        &self.test_set
    }

    pub fn features_flattened(set: ArrayView3<'_, T>) -> ArrayView2<'_, T> {
        let length = set.dim().0;
        set.into_shape_with_order((length, 28 * 28)).unwrap()
    }

    pub fn targets_unrolled(set: ArrayView1<'_, T>) -> Array2<T>
    where
        T: Zero + Clone + ToPrimitive + One,
    {
        let length = set.dim();
        let mut result = Array2::zeros((length, 10));
        Zip::from(result.rows_mut())
            .and(&set)
            .for_each(|mut row, y| {
                row[T::to_usize(y).unwrap()] = T::one();
            });
        result
    }

    pub fn to_owned(self) -> Mnist<'static, T>
    where
        T: Clone,
    {
        Mnist {
            train_set: self.train_set.to_owned(),
            test_set: self.test_set.to_owned(),
        }
    }

    pub fn from_parts(
        train_set: MnistSet<'a, T>,
        test_set: MnistSet<'a, T>
    ) -> Self {
        Self {
            train_set,
            test_set
        }
    }
    
    pub fn load(path: impl AsRef<Path>) -> Result<Mnist<'static, T>, PyErr>
    where
        T: Clone,
    {
        Python::initialize();
        let mnist = Python::attach(|py| {
            let dataset = Datasets::new(py, path)?;
            Ok::<_, PyErr>(dataset.load_as_mnist()?.to_owned())
        })?;

        Ok(mnist)
    }
}

impl<'py, T: Element, D: Dimension> ArrayStorage<'py, T, D> {
    fn view(&self) -> ArrayView<'_, T, D> {
        match self {
            ArrayStorage::Python(x) => x.as_array(),
            ArrayStorage::Rust(x) => x.view(),
        }
    }

    fn to_owned(self) -> ArrayStorage<'static, T, D>
    where
        T: Clone,
    {
        match self {
            ArrayStorage::Python(x) => ArrayStorage::Rust(x.to_owned_array()),
            ArrayStorage::Rust(x) => ArrayStorage::Rust(x.to_owned()),
        }
    }
}

impl<DF: Dimension, DT: Dimension, T: Element> Set<'_, DF, DT, T> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn length(&self) -> usize {
        self.num_rows
    }

    pub fn targets(&self) -> ArrayView<'_, T, DT> {
        self.targets.view()
    }

    pub fn features(&self) -> ArrayView<'_, T, DF> {
        self.features.view()
    }

    pub fn to_owned(self) -> Set<'static, DF, DT, T>
    where
        T: Clone,
    {
        Set {
            name: self.name.to_owned(),
            features: self.features.to_owned(),
            targets: self.targets.to_owned(),
            num_rows: self.num_rows,
        }
    }
}

pub struct Datasets<'py>(HashMap<String, Dataset<'py>>);

impl<'py> Datasets<'py> {
    pub fn new(py: Python<'py>, path: impl AsRef<Path>) -> Result<Self, PyErr> {
        let datasets = py.import("datasets")?;
        let path = path.as_ref();
        let dataset = datasets.call_method1("load_from_disk", (path,))?;

        let mut this = Datasets(HashMap::new());
        debug!("Found datasets: {}", dataset);
        for name in dataset.try_iter()? {
            let name = name?.extract::<String>()?;
            let item = dataset.get_item(&name)?;
            let num_rows = item.getattr("num_rows")?.extract::<usize>()?;

            this.0.insert(
                name,
                Dataset {
                    num_rows,
                    features: item,
                },
            );
        }

        Ok(this)
    }

    pub fn load_as_mnist<T>(&self) -> Result<Mnist<'_, T>, PyErr>
    where
        T: Element,
    {
        let train = &self.0["train"];
        let test = &self.0["test"];

        fn load_all_features_as_ndarray<'py, T, D>(
            obj: &Bound<'py, PyAny>,
            feature_name: &str,
        ) -> Result<ArrayStorage<'py, T, D>, PyErr>
        where
            T: Element,
            D: Dimension,
        {
            let features = obj.get_item(feature_name)?;
            let py: Python = obj.py();
            let numpy = py.import("numpy")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", dtype::<T>(py))?;

            let features = numpy
                .getattr("array")?
                .call((features,), Some(&kwargs))?
                .extract::<PyReadonlyArray<T, D>>()?;
            Ok(ArrayStorage::Python(features.readonly()))
        }

        Ok(Mnist {
            train_set: Set {
                name: String::from("train"),
                num_rows: train.num_rows,
                targets: load_all_features_as_ndarray(&train.features, "label")?,
                features: load_all_features_as_ndarray(&train.features, "image")?,
            },
            test_set: Set {
                name: String::from("test"),
                num_rows: test.num_rows,
                targets: load_all_features_as_ndarray(&test.features, "label")?,
                features: load_all_features_as_ndarray(&test.features, "image")?,
            },
        })
    }
}
