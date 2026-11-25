# Hit Song Science

**Predicting whether a song will chart, from its audio features, with a linear SVM served through a web app.**

---

## About

This project trains a **linear Support Vector Machine (SVM)**, written from scratch with NumPy and trained with the **Pegasos** algorithm, to predict whether a song is **Popular** (it stays on the chart for at least _N_ weeks) or **Not popular** (it never charts).

It has three parts:

| Part                  | What it does                                                                                     | Main files                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| **Model development** | Loads the datasets, tunes and trains the SVM, compares feature sets, chart windows and periods   | `experiment.ipynb`, `preprocessing.py`, `feature_extraction.py`, `model.py`, `evaluation.py`, `config.py` |
| **Backend**           | FastAPI server that loads the trained model and predicts from an uploaded song clip              | `backend/`                                                                                                |
| **Frontend**          | Vue web app to upload a clip, play it back and see the prediction with its confidence            | `frontend/`                                                                                               |

## Dataset

The datasets are **included** in this repository, in `data/`. Each row is one song with:

- **Spotify-style audio features**: `explicit`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- **MFCC statistics**: mean and standard deviation over time of 13 MFCCs (26 values), computed from an audio segment of the song
- **Chart information**: `weeks_charted` and the label `is_trending`
- Metadata (artist, track name, genre, release date, ...), not used as features

### Labels

For an observation window of **_N_ weeks** (N = 1, 2, 3, 4, 8):

| Label           | `is_trending` | Rule                    |
| --------------- | ------------- | ----------------------- |
| **Popular**     | `1`           | `weeks_charted >= N`    |
| **Not popular** | `-1`          | `weeks_charted == 0`    |

Songs in between (charted, but for fewer than _N_ weeks) are dropped. Every dataset is **balanced** (as many Popular as Not popular songs).

### Dataset variants

Files are named `data/<variant>/dataset_n{N}_{subset}.csv`:

| Dimension                 | Values                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Missing-value variant** | `keep_all`, `no_empty_genre`, `no_missing_date`, `complete_only` (empty genre and missing date both dropped) |
| **Window _N_**            | `n1`, `n2`, `n3`, `n4`, `n8`                                                                                 |
| **Release subset**        | `all`, `old` (released before 2021), `new` (released 2021 or later)                                          |

The experiments use `data/complete_only`. Its sizes:

| Window | all   | old   | new   |
| ------ | ----- | ----- | ----- |
| n1     | 2,448 | 1,036 | 1,412 |
| n2     | 2,174 | 962   | 1,212 |
| n3     | 2,030 | 910   | 1,120 |
| n4     | 1,902 | 838   | 1,064 |
| n8     | 1,588 | 720   | 868   |

`data/scenario_summary.csv` lists how many songs every rule removed for each dataset, and `data/base/` holds the base dataset of each variant.

## Model

| Model         | Type                                                                          |
| ------------- | ----------------------------------------------------------------------------- |
| **BinarySVM** | Linear soft-margin SVM trained with Pegasos (primal sub-gradient descent)     |

The SVM is wrapped in `Standardized`, which fits a z-score scaler on the training data only. No scikit-learn is used: the SVM, the stratified split, k-fold cross-validation, grid search and the metrics are all implemented in this project.

### Experiment protocol

For every scenario, `experiment.ipynb`:

1. Splits the dataset **80:20** (stratified, seed 42). The 20% test set is not touched until step 4.
2. Tunes `C` with **5-fold cross-validation** on the 80% training set.
3. Chooses the best feature set / window by this CV accuracy.
4. Retrains on the full 80% with the best `C` and reports **accuracy** on the 20% test set.

| Stage | Question                                     | Data                                     |
| ----- | -------------------------------------------- | ---------------------------------------- |
| 1     | Spotify vs MFCC vs Spotify + MFCC features   | `n1_all`                                 |
| 2     | Best observation window: n1, n2, n3, n4, n8  | `nX_all`, best feature set               |
| 3     | Does old data still predict new songs?       | `nX_old`, `nX_new` (best window)         |

### Results

**Stage 1: feature sets** (`n1_all`)

| Feature set        | Features | Best C | CV accuracy | Test accuracy |
| ------------------ | -------- | ------ | ----------- | ------------- |
| Spotify            | 10       | 1      | 0.642       | 0.647         |
| MFCC               | 26       | 0.01   | 0.525       | 0.527         |
| **Spotify + MFCC** | 36       | 0.1    | **0.661**   | **0.659**     |

**Stage 2: observation windows** (Spotify + MFCC)

| Window | Songs | Best C | CV accuracy | Test accuracy |
| ------ | ----- | ------ | ----------- | ------------- |
| n1     | 2,448 | 0.1    | 0.661       | 0.659         |
| n2     | 2,174 | 1      | 0.673       | 0.673         |
| n3     | 2,030 | 0.1    | 0.662       | 0.668         |
| n4     | 1,902 | 0.1    | 0.655       | 0.682         |
| **n8** | 1,588 | 1      | **0.684**   | 0.626         |

**Stage 3: old vs new songs** (n8, Spotify + MFCC)

| Scenario  | Train   | Test    | Test accuracy |
| --------- | ------- | ------- | ------------- |
| Old → old | old 80% | old 20% | 0.667         |
| New → new | new 80% | new 20% | 0.713         |
| Old → new | old 80% | new 20% | 0.678         |

Training on recent songs instead of old ones gains about **3.4 percentage points** on the same new test songs.

The web app uses the Stage 2 **n8, Spotify + MFCC** model (`models/svm_n8_spotify_mfcc.pkl`).

## Project structure

```
hit-song-science/
├── backend/                 FastAPI server
│   ├── server.py            app, CORS, model loading
│   ├── routers/             /api/health, /api/predict
│   ├── services/            feature building + prediction, ReccoBeats client
│   ├── schemas/             request / response models
│   └── utils/               upload validation, error codes
├── frontend/                Vue 3 + Vite + Tailwind web app
├── data/                    datasets (see "Dataset")
├── models/                  trained model (.pkl) + its metadata (.json)
├── requirements/            Python dependencies, split by part
├── config.py                shared settings (features, C grid, seed, paths, ...)
├── preprocessing.py         dataset loading, stratified split, z-score scaling
├── feature_extraction.py    MFCC statistics of one clip
├── model.py                 BinarySVM (Pegasos), Standardized wrapper, save / load
├── evaluation.py            cross-validation, grid search, metrics and plots
├── experiment.ipynb         the full training and evaluation pipeline
└── requirements.txt         installs every Python dependency
```

## Getting started

### Prerequisites

- **Python 3.11** (conda or a virtual environment)
- **Node.js** `22.18+` or `24.12+` (only for the web app)
- An internet connection for the web app (the backend calls the ReccoBeats API)

### 1. Get the project

```bash
git clone https://github.com/GDnichoTP/hit-song-science.git
cd hit-song-science
```

Or download the ZIP file from GitHub and extract it.

### 2. Create a Python environment

With conda:

```bash
conda create -n hit-song python=3.11
conda activate hit-song
```

Or with `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

### 3. Install the Python dependencies

```bash
pip install -r requirements.txt
```

This installs everything. To install only one part, use a file in `requirements/`:

| File                       | Installs                                             |
| -------------------------- | ---------------------------------------------------- |
| `requirements/base.txt`    | Shared core (numpy, pandas, librosa, soundfile)      |
| `requirements/dev.txt`     | Core + notebook, evaluation and plotting tools       |
| `requirements/backend.txt` | Core + FastAPI server and the ReccoBeats client      |

### 4. Train the model (optional)

The trained model is already in `models/`. To reproduce the experiments or retrain it:

1. Open `experiment.ipynb`
2. Select the Python environment from step 2 as the kernel
3. Run all cells

The last cell saves `models/svm_n8_spotify_mfcc.pkl` and `models/svm_n8_spotify_mfcc.json` (feature order, accuracy and the `platt_a` slope used for the confidence). The backend needs **both** files.

## Running the web app

Run the backend and the frontend in two terminals.

### Backend

From the project root, with the Python environment active:

```bash
uvicorn backend.server:app --reload
```

The API runs at <http://localhost:8000>, with interactive docs at <http://localhost:8000/docs>.

| Endpoint            | Description                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| `GET /api/health`   | Whether the model is loaded, its name and the maximum clip length                    |
| `POST /api/predict` | Predict one uploaded `file` (`.wav` or `.mp3`), with an optional `explicit` flag     |

Uploads must be at most **15 MB** and between **10 and 30 seconds** long.

#### How a prediction is made

For each upload, the backend builds the same 36 features the model was trained on:

| Features            | Source                                                                                    |
| ------------------- | ----------------------------------------------------------------------------------------- |
| `explicit`          | The checkbox in the web app (it cannot be heard in the audio)                             |
| 9 audio features    | [ReccoBeats](https://reccobeats.com) audio-features API, from the uploaded clip           |
| 26 MFCC statistics  | Computed locally with librosa, the same way as the training data                          |

Spotify's own audio-features endpoint is closed to new apps, so ReccoBeats is used instead. It is free and needs no API key, but it can be rate-limited or unavailable; the app then shows a matching error. **The uploaded clip is sent to ReccoBeats** to compute these features.

The sign of the SVM score gives the label, and a sigmoid of the score (slope `platt_a`) gives the confidence.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The green dot in the header shows that the frontend can reach the backend and the model is loaded.

> **Different port or host?** The backend only accepts requests from `http://localhost:5173` and `http://127.0.0.1:5173`. If Vite starts on another port, add that origin to `CORS_ORIGINS` in `backend/server.py`. To point the frontend at a backend that is not on `http://localhost:8000`, create `frontend/.env` with `VITE_API_URL=http://your-host:port`.

### Using the app

1. Drag a `.wav` or `.mp3` clip of a song (**10 to 30 seconds**) into the upload panel, or click **choose a file**
2. Tick **The song has explicit lyrics** if it is marked "E" on streaming services
3. Optionally play it back to check it
4. Click **Predict**

The model always answers Popular or Not popular. The served model was right on about **63%** of its test songs, and it only hears the clip and the explicit flag (not the artist, promotion or release timing), so treat the result as a rough estimate, not a guarantee.

## References

- Shalev-Shwartz, S., Singer, Y., Srebro, N., Cotter, A. _Pegasos: Primal Estimated sub-GrAdient SOlver for SVM._ Mathematical Programming, 2011.
- [ReccoBeats API](https://reccobeats.com)
- [librosa](https://librosa.org) for audio loading and MFCC extraction
