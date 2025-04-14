# ffn-polars

[![codecov](https://codecov.io/gh/rwspielman/ffn-polars/graph/badge.svg?token=RDWPURUB3K)](https://codecov.io/gh/rwspielman/ffn-polars)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📘 [View the Documentation](https://rwspielman.github.io/ffn-polars/)

🚀 A blazing-fast, Polars-powered reimplementation of [`ffn`](https://github.com/pmorissette/ffn) for portfolio analytics and performance measurement.

## ✨ What is this?

`ffn-polars` is a modern rewrite of the original `ffn` library using [Polars](https://pola.rs), a Rust-backed DataFrame engine designed for performance and scalability.

> ✅ Faster than pandas  
> ✅ Lazy evaluation + multithreaded execution  
> ✅ Memory-efficient on large datasets  
> ✅ Same core metrics: returns, CAGR, Sharpe, drawdowns, volatility, etc.

---

## 📦 Installation (Coming Soon)

```bash
pip install ffn-polars
```

## 🤝 Contributing

Contributions welcome! Open an issue or PR if you’d like to improve a function or add a new metric.

## 📄 License

MIT. Originally inspired by [`ffn`](https://github.com/pmorissette/ffn) by Pierre R. Morissette.

## TODOs

- Add more tests
- Add more support for nanosecond timestamps

### Tick Data Ideas

#### Microstructure Metrics

- Trade Rate
- Inter-Trade Time
- Quote-to-Trade Ratio
- Volume Profile
- Tick Imbalance
- Price Impact

#### Volatility and Returns (High-Frequency)

- Realized Volatility
- Micro Returns
- Rolling Volatility
- OHLC Bars
- Garman-Klass / Parkinson

#### Latency / Time-dynamics

- Burst Detection
- Idle time
- Trade Clustering
- Anomaly Detection

#### Books to pull from

- "Advanced Portfolio Management" by Giuseppe A. Paleologo
- "Handbook of Price Impact Modeling" by Kevin T Webster
- "Advances in Financial Machine Learning" by Marcos Lopez de Prado
