use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_polars::PySeries;
use polars::prelude::*;
use statrs::distribution::{StudentsT, ContinuousCDF};

#[pyfunction]
fn prob_mom(a: PySeries, b: PySeries) -> PyResult<PySeries> {
    let a = a
        .0
        .f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let b = b
        .0
        .f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let diffs: Vec<f64> = a
        .into_iter()
        .zip(b)
        .filter_map(|(a, b)| Some(a? - b?))
        .collect();

    let n = diffs.len();
    let result = if n < 2 {
        Series::new("prob_momentum".into(), &[None::<f64>])
    } else {
        let mean = diffs.iter().sum::<f64>() / n as f64;
        let std = (diffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();

        if std == 0.0 {
            Series::new("prob_momentum".into(), &[None::<f64>])
        } else {
            let ir = mean / std;
            let t_dist = StudentsT::new(0.0, 1.0, (n - 1) as f64).unwrap();
            let p = t_dist.cdf(ir);
            Series::new("prob_momentum".into(), &[p])
        }
    };

    Ok(PySeries(result))
}

#[pymodule]
fn _rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prob_mom, m)?)?;
    Ok(())
}



