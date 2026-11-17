# 🎧 Podcast Episode Popularity Prediction

Predict how many **minutes** a listener will spend on a podcast episode, using only information known **before** the episode is published (length, host and guest popularity, ads, genre, release day and time).

Built on the Kaggle Playground Series **S5E4 – Predict Podcast Listening Time** dataset (750,000 episodes) with three regression models and an optional pretrained **Hugging Face** sentiment feature. Includes a simple **Streamlit** app for live predictions.

> B.Tech Information Technology · Rai Technology University, Bengaluru

---

## Why this project?

Creators guess how long an episode should be, how many ads to include, and when to release it. Existing research mostly classifies whole podcasts as "preferred / not preferred" or uses expensive transcripts. This project instead:

- predicts a **number of minutes** (regression), not a label
- works **per episode**, not per show
- needs **metadata only**, so it works before publishing
- tests a **pretrained language model as a feature** for classical models
- checks whether the model simply rewards famous hosts (**popularity bias**)

## Results

Evaluated on 150,000 episodes the models never saw (80/20 split, full 750,000-row dataset).

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| Linear Regression | 13.34 | 9.76 | 0.758 |
| **Random Forest Regressor** | **12.97** | **9.41** | **0.771** |
| Gradient Boosting (Histogram-based) | 13.05 | 9.51 | 0.768 |

**Key findings**

- Random Forest is the best model and explains about 77% of the variation in listening time.
- **Episode length** drives about 95% of Random Forest's decisions (importance 0.948).
- **No popularity bias:** host and guest popularity features hold only 2.0% of the importance.
- Error is similar across genres: mean absolute error ranges from 9.23 (Sports) to 9.57 (Music) minutes.
- The Hugging Face sentiment feature gave **no measurable gain** (RMSE 12.98 without, 12.97 with). Every title in the dataset is a generic label such as `Episode 98` (only 100 unique titles in 750,000 rows), so there is no real sentiment to detect.

## Repository structure

```
.
├── app.py             # Streamlit app: train models and predict a new episode
├── main.ipynb         # Full notebook: EDA, features, Hugging Face, models, analysis
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

The dataset (`train.csv`) is **not** included because of its size. Download it from Kaggle (see below).

## Dataset

[Kaggle Playground Series S5E4 – Predict Podcast Listening Time](https://www.kaggle.com/competitions/playground-series-s5e4)

| Column | Description |
| --- | --- |
| `id` | Row number (dropped) |
| `Podcast_Name` | Name of the show |
| `Episode_Title` | Title text (e.g. `Episode 98`) |
| `Episode_Length_minutes` | Length of the episode |
| `Genre` | Comedy, Technology, News, etc. |
| `Host_Popularity_percentage` | Popularity of the host |
| `Guest_Popularity_percentage` | Popularity of the guest |
| `Publication_Day` | Day of the week |
| `Publication_Time` | Morning / Afternoon / Evening / Night |
| `Number_of_Ads` | Number of ads in the episode |
| `Episode_Sentiment` | Positive / Neutral / Negative |
| `Listening_Time_minutes` | **Target**: minutes listened |

Playground Series datasets are typically artificially generated, so patterns may differ from a real podcast platform.

## Pipeline

1. **Clean**: drop `id` and duplicates; fill missing numbers with the median and missing text with the most frequent value (about 19.5% of guest popularity and 11.6% of episode length were missing).
2. **Engineer features**
   - `Ads_per_minute`: ad density instead of raw ad count
   - `Is_Weekend`: 1 for Saturday/Sunday
   - `Popularity_Gap`: host popularity minus guest popularity
   - `HF_Sentiment_Score`: Hugging Face sentiment of the title (notebook only)
3. **Encode** text columns with `LabelEncoder`.
4. **Split** 80/20 and **scale** with `StandardScaler` (fitted on training data only to avoid leakage).
5. **Train** Linear Regression, Random Forest (150 trees, depth 14) and histogram Gradient Boosting (200 iterations, depth 8, learning rate 0.08).
6. **Evaluate** with RMSE, MAE and R², then run feature importance, an ablation study, genre-wise error and a popularity-bias check.

### Why histogram Gradient Boosting?

scikit-learn's standard `GradientBoostingRegressor` builds trees strictly one after another with no parallelism and was far too slow on 750,000 rows. `HistGradientBoostingRegressor` bins feature values (the same idea behind LightGBM), so it trains much faster while still being gradient boosting.

## Hugging Face sentiment feature

The notebook uses the pretrained model [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) through the `transformers` pipeline to score each episode title as positive or negative. The result is turned into one signed number between -1 and +1.

Because the dataset has only **100 unique titles**, the model scores those 100 titles once and maps the results back to all 750,000 rows. That is about **7,500× fewer model calls** with identical results.

Hugging Face is used in the notebook only. The Streamlit app is the lightweight prediction tool and does not need `torch` or `transformers`.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

1. In the sidebar choose **Demo data** (instant, no file needed) or **Upload CSV** and select the Kaggle `train.csv`.
2. Set **Max rows to train on**. The default is 50,000 for quick training. Increase it for results closer to the notebook.
3. Click **Train models** to see the RMSE / MAE / R² comparison.
4. Enter an episode's details and click **Predict** to get the predicted listening time. You can compare all three models' predictions.

How predictions stay correct: derived features (`Ads_per_minute`, `Is_Weekend`, `Popularity_Gap`) are built automatically, categories are encoded with the saved training encoders, values are scaled with the saved scaler, and the result is limited to between 0 and the episode length.

> **Note:** *Demo data* is synthetic and only for trying out the app. Real results come from the Kaggle dataset. The app trains on a sample by default and has no Hugging Face feature, so its numbers will be close to, but not identical to, the notebook's.

## Run the notebook

1. Download `train.csv` from Kaggle and place it next to `main.ipynb`.
2. To run the Hugging Face section, install the optional packages (a GPU is used automatically if available, otherwise the CPU):

   ```bash
   pip install torch transformers
   ```

3. Open the notebook and run all cells:

   ```bash
   jupyter notebook main.ipynb
   ```

If `torch` or `transformers` are missing, the sentiment step is skipped with a message and the rest of the notebook still runs.

## Limitations

- The dataset is likely synthetic, and listening time is a proxy for popularity, not total streams.
- Episode titles are generic, so the Hugging Face feature carries no real signal here.
- Text columns use label encoding, which implies an order between categories that does not exist.
- A single train/test split is used, with no cross-validation or hyperparameter search.
- The app cannot predict for podcast names that were not in the training data.

## Future work

- Cross-validation and hyperparameter tuning
- One-hot or target encoding for nominal columns
- Real episode descriptions or titles for the Hugging Face feature
- Audio-based features
- Handling unseen categories in the app
- Testing on data from a real podcast platform
