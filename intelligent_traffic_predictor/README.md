# Intelligent Traffic Predictor

A Python + Streamlit web application for exploring traffic data, training a machine-learning model, estimating traffic volume/congestion, and exporting reports.

## Features
- Overview dashboard with traffic KPIs and charts
- CSV upload and dataset inspection
- Random Forest regression or classification
- Model evaluation: MAE, RMSE, R² or Accuracy, Precision, Recall, F1
- Interactive prediction form
- Peak-hour, day-wise and weather-wise analysis
- CSV export of datasets and evaluation results

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

The app opens in your browser, usually at `http://localhost:8501`.

## Included dataset
`data/sample_traffic_data.csv` is a synthetic demonstration dataset generated for this project. It is not official traffic sensor data and must not be presented as real measurements.

## Expected columns for included workflow
The sample dataset contains:
- `hour`: hour of day (0–23)
- `day`: weekday/weekend
- `weather`: weather condition
- `temperature`: illustrative temperature
- `is_holiday`: holiday indicator
- `road_occupancy`: illustrative road occupancy percentage
- `traffic_volume`: vehicles per hour (prediction target)
- `congestion_level`: low/moderate/high category

To train a volume model, select `traffic_volume` and Regression. To predict a category, select `congestion_level` and Classification.

## Important limitations
- This prototype does not fetch live traffic data.
- It does not provide real GPS routes or turn-by-turn navigation.
- It does not control traffic signals.
- Predictions are only as reliable as the dataset used for training.
- The included sample data is synthetic and intended for software demonstration only.

## Suggested future enhancements
1. Connect an approved live traffic API or city traffic sensor feed.
2. Add road-segment IDs, latitude/longitude and map visualization.
3. Train and compare multiple algorithms (Decision Tree, Random Forest, Gradient Boosting, LSTM where justified).
4. Add time-series forecasting and cross-validation.
5. Deploy with authentication, logging, privacy controls and model monitoring.
