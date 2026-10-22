# ML4T Chapter 2 Assignment Solutions

This package contains complete solutions to the five Chapter 2 exercises from **The Financial Data Universe**:

1. US Equities EDA
2. Corporate Actions
3. Point-in-Time Validation
4. Survivorship Bias Detection
5. File Storage Benchmark

## Main file

- `ML4T_Chapter02_Assignment_Solution.ipynb`
- `assignment_support.py` contains the reusable loading, validation, portfolio, and benchmark helpers.
- `requirements.txt` lists the Python dependencies needed for a clean rerun.

## How to run

The submitted notebook already contains saved outputs. To rerun it, download the public WIKI Prices data set used by the official *Machine Learning for Trading, 3rd Edition* repository:

<https://github.com/stefan-jansen/machine-learning-for-trading>

1. Create a Python environment and run `pip install -r requirements.txt`.
2. Download and extract `WIKI_PRICES.csv` from the public WIKI Prices source documented by the ML4T project.
3. Set `WIKI_DATA_PATH` to the extracted CSV or a converted Parquet file.
4. Open the notebook from this folder and select **Restart Kernel and Run All Cells**.

Example on macOS or Linux:

```bash
export WIKI_DATA_PATH=/absolute/path/to/WIKI_PRICES.csv
jupyter lab ML4T_Chapter02_Assignment_Solution.ipynb
```

## Result notes

- Exercises 1, 2, and 4 use the Quandl WIKI panel. The data file is excluded from Git because the extracted CSV is approximately 1.8 GB.
- Exercise 3 uses simulated data with a fixed random seed, making its output reproducible.
- Exercise 5 generates a deterministic one-million-row synthetic OHLCV panel. Read and write times depend on the machine, storage device, and page cache.
- Quantitative conclusions are printed from measured results. The notebook does not assign a predetermined winner or hard-code performance figures.

## Attribution

The assignment structure and methodology are based on Stefan Jansen's ML4T repository, released under the MIT License.

