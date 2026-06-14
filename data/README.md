# Dataset

This project uses the FD001 subset of NASA's C-MAPSS turbofan degradation
dataset. The data files are not committed to the repo, so you need to download
them once before training.

## Files you need

Put these three files in `data/raw/`:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

## Where to get them

The dataset comes from NASA's Prognostics Center of Excellence. A mirror that
hosts the plain-text files is the `CMAPSSData` folder here:

https://github.com/hankroark/Turbofan-Engine-Degradation

Run this from the project root to download the three files straight into
`data\raw\`:

```powershell
$base = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"
foreach ($f in "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt") {
    Invoke-WebRequest -Uri ($base + $f) -OutFile "data\raw\$f"
}
```

## File format

The train and test files are whitespace-separated with no header. Columns, in
order:

1. unit number
2. cycle
3-5. three operational settings
6-26. sensors 1 through 21

In the training file each engine runs until it fails. In the test file each
engine stops some cycles before failure, and `RUL_FD001.txt` lists the true
remaining cycles for each test engine, one value per line.
