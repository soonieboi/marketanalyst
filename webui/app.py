import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
from zoneinfo import ZoneInfo
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def make_us_market_timestamps(last_timestamp, periods, time_diff):
    """Generate future regular-session US equity timestamps."""
    eastern = ZoneInfo("America/New_York")
    start_time = datetime.time(9, 30)
    end_time = datetime.time(16, 0)
    step = pd.Timedelta(time_diff)

    current = pd.Timestamp(last_timestamp)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    current = current.tz_convert(eastern)

    timestamps = []
    candidate = current + step
    while len(timestamps) < periods:
        session_start = candidate.replace(hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0)
        session_end = candidate.replace(hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0)

        if candidate.weekday() >= 5:
            days_ahead = 7 - candidate.weekday()
            candidate = (candidate + pd.Timedelta(days=days_ahead)).replace(
                hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0
            )
            continue

        if candidate < session_start:
            candidate = session_start
            continue

        if candidate > session_end:
            candidate = (candidate + pd.Timedelta(days=1)).replace(
                hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0
            )
            continue

        timestamps.append(candidate.tz_convert("UTC"))
        candidate = candidate + step

    return pd.Series(pd.DatetimeIndex(timestamps), name='timestamps')

def normalize_start_datetime(start_date, timestamp_series):
    """Match browser-provided start_date timezone to the loaded data."""
    start_dt = pd.Timestamp(start_date)
    data_tz = getattr(timestamp_series.dt, "tz", None)

    if data_tz is not None:
        if start_dt.tzinfo is None:
            return start_dt.tz_localize(data_tz)
        return start_dt.tz_convert(data_tz)

    if start_dt.tzinfo is not None:
        return start_dt.tz_convert("UTC").tz_localize(None)
    return start_dt

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def chart_times(values):
    return [pd.Timestamp(value).isoformat() for value in values]

def chart_times_et(values):
    eastern = ZoneInfo("America/New_York")
    times = []
    for value in values:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        times.append(ts.tz_convert(eastern).strftime("%Y-%m-%d %H:%M"))
    return times

def chart_numbers(values):
    return [float(value) for value in values]

def calculate_atr(df, window=14):
    if df is None or len(df) < 2:
        return 0.0

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.tail(min(window, len(true_range))).mean())

def calculate_close_mae(pred_df, actual_df):
    if pred_df is None or actual_df is None or len(pred_df) == 0 or len(actual_df) == 0:
        return None

    length = min(len(pred_df), len(actual_df))
    pred_close = pred_df['close'].iloc[:length].astype(float).to_numpy()
    actual_close = actual_df['close'].iloc[:length].astype(float).to_numpy()
    return float(np.mean(np.abs(pred_close - actual_close)))

def round_price(value):
    return None if value is None else round(float(value), 4)

def compute_trade_overlay(input_df, pred_df, actual_df=None):
    """Create conservative decision-support zones from the forecast path."""
    if input_df is None or pred_df is None or len(input_df) == 0 or len(pred_df) == 0:
        return None

    current_close = float(input_df['close'].iloc[-1])
    forecast_close = float(pred_df['close'].iloc[-1])
    predicted_high = float(pred_df['high'].max())
    predicted_low = float(pred_df['low'].min())
    recent_window = input_df.tail(min(60, len(input_df)))
    recent_support = float(recent_window['low'].min())
    recent_resistance = float(recent_window['high'].max())
    atr = calculate_atr(input_df)
    if atr <= 0:
        atr = max(current_close * 0.005, 0.01)

    close_mae = calculate_close_mae(pred_df, actual_df)
    error_buffer = close_mae if close_mae is not None else max(atr * 0.5, current_close * 0.0025)
    forecast_move_pct = ((forecast_close - current_close) / current_close) * 100
    threshold_pct = max((atr / current_close) * 25, 0.3)

    if forecast_move_pct > threshold_pct:
        bias = "bullish"
    elif forecast_move_pct < -threshold_pct:
        bias = "bearish"
    else:
        bias = "neutral"

    setup = {
        'bias': bias,
        'current_close': round_price(current_close),
        'forecast_close': round_price(forecast_close),
        'forecast_move_pct': round(forecast_move_pct, 2),
        'predicted_high': round_price(predicted_high),
        'predicted_low': round_price(predicted_low),
        'recent_support': round_price(recent_support),
        'recent_resistance': round_price(recent_resistance),
        'atr': round_price(atr),
        'error_buffer': round_price(error_buffer),
        'error_source': 'comparison window close MAE' if close_mae is not None else 'estimated from recent volatility',
        'entry_zone': None,
        'stop_zone': None,
        'target_zone': None,
        'risk_reward': None,
        'confidence': 'low',
        'notes': []
    }

    if bias == "neutral":
        setup['notes'].append("No clear directional edge from this forecast. Treat as a watch-only setup.")
        return setup

    if bias == "bullish":
        entry_low = current_close - 0.75 * atr
        entry_high = current_close + 0.25 * atr
        stop = entry_low - atr
        target_low = max(current_close, forecast_close - error_buffer)
        target_high = predicted_high - error_buffer
        entry_mid = (entry_low + entry_high) / 2
        risk = entry_mid - stop
        reward = target_low - entry_mid
        if target_high <= entry_mid or reward <= 0 or risk <= 0:
            setup['notes'].append("Forecast leans bullish, but the adjusted target does not justify a clean long zone.")
        setup['entry_zone'] = {'low': round_price(entry_low), 'high': round_price(entry_high)}
        setup['stop_zone'] = {'price': round_price(stop)}
        setup['target_zone'] = {'low': round_price(target_low), 'high': round_price(max(target_low, target_high))}
    else:
        entry_low = current_close - 0.25 * atr
        entry_high = current_close + 0.75 * atr
        stop = entry_high + atr
        target_low = predicted_low + error_buffer
        target_high = min(current_close, forecast_close + error_buffer)
        entry_mid = (entry_low + entry_high) / 2
        risk = stop - entry_mid
        reward = entry_mid - target_high
        if target_low >= entry_mid or reward <= 0 or risk <= 0:
            setup['notes'].append("Forecast leans bearish, but the adjusted target does not justify a clean short zone.")
        setup['entry_zone'] = {'low': round_price(entry_low), 'high': round_price(entry_high)}
        setup['stop_zone'] = {'price': round_price(stop)}
        setup['target_zone'] = {'low': round_price(min(target_low, target_high)), 'high': round_price(target_high)}

    if risk > 0:
        rr = reward / risk
        setup['risk_reward'] = round(float(rr), 2)
        if rr >= 1.5 and abs(forecast_move_pct) >= threshold_pct * 2:
            setup['confidence'] = 'medium'
        if rr < 1:
            setup['notes'].append("Risk/reward is below 1.0 after the error buffer.")

    setup['notes'].append("Use these as zones for review, not automatic trade instructions.")
    return setup

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=chart_times_et(historical_df['timestamps']) if 'timestamps' in historical_df.columns else list(historical_df.index),
        open=chart_numbers(historical_df['open']),
        high=chart_numbers(historical_df['high']),
        low=chart_numbers(historical_df['low']),
        close=chart_numbers(historical_df['close']),
        name=f'Historical candles ({len(historical_df)} points)',
        increasing_line_color='#94A3B8',
        decreasing_line_color='#64748B',
        increasing_fillcolor='rgba(148, 163, 184, 0.35)',
        decreasing_fillcolor='rgba(100, 116, 139, 0.35)',
        opacity=0.45
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Prefer the predictor's timestamp index. In latest-data mode this is the
        # next regular US market-session schedule; in comparison mode it is the
        # selected historical target window.
        try:
            pred_timestamps = pd.to_datetime(pred_df.index)
        except Exception:
            pred_timestamps = None

        if pred_timestamps is None or len(pred_timestamps) != len(pred_df):
            if 'timestamps' in df.columns and len(historical_df) > 0:
                last_timestamp = historical_df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                pred_timestamps = make_us_market_timestamps(last_timestamp, len(pred_df), time_diff)
            else:
                pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=chart_times_et(pred_timestamps),
            open=chart_numbers(pred_df['open']),
            high=chart_numbers(pred_df['high']),
            low=chart_numbers(pred_df['low']),
            close=chart_numbers(pred_df['close']),
            name='Predicted candles',
            increasing_line_color='#2563EB',
            decreasing_line_color='#2563EB',
            increasing_fillcolor='rgba(37, 99, 235, 0.16)',
            decreasing_fillcolor='rgba(37, 99, 235, 0.16)',
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=chart_times_et(pred_timestamps),
            y=chart_numbers(pred_df['close']),
            mode='lines',
            name='Predicted close',
            line={'color': '#1D4ED8', 'width': 3, 'dash': 'dash'}
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=chart_times_et(actual_timestamps),
            open=chart_numbers(actual_df['open']),
            high=chart_numbers(actual_df['high']),
            low=chart_numbers(actual_df['low']),
            close=chart_numbers(actual_df['close']),
            name='Actual candles',
            increasing_line_color='#F97316',
            decreasing_line_color='#F97316',
            increasing_fillcolor='rgba(249, 115, 22, 0.18)',
            decreasing_fillcolor='rgba(249, 115, 22, 0.18)',
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=chart_times_et(actual_timestamps),
            y=chart_numbers(actual_df['close']),
            mode='lines',
            name='Actual close',
            line={'color': '#EA580C', 'width': 3}
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Forecast Comparison - blue dashed = predicted close, orange solid = actual close',
        xaxis_title='US Eastern Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[chart_times_et([all_timestamps[0]])[0], chart_times_et([all_timestamps[-1]])[0]],
                rangeslider_visible=False,
                type='category'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = normalize_start_datetime(start_date, df['timestamps'])
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use the latest completed candles and forecast forward from the
                    # newest timestamp. This is the live-reference path.
                    x_df = df.iloc[-lookback:][required_cols]
                    x_timestamp = df.iloc[-lookback:]['timestamps']
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    y_timestamp = make_us_market_timestamps(df['timestamps'].iloc[-1], pred_len, time_diff)
                    prediction_type = "Kronos model prediction (latest completed data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = normalize_start_datetime(start_date, df['timestamps'])
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # Latest-data mode forecasts beyond the newest candle, so actual future
            # candles are not available yet. Historical comparison is handled by
            # selecting a custom start_date window.
            actual_df = None
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = normalize_start_datetime(start_date, df['timestamps'])
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: show the most recent lookback window.
            historical_start_idx = max(0, len(df) - lookback)
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date:
                # Custom time period: use selected window data to calculate timestamps
                start_dt = normalize_start_datetime(start_date, df['timestamps'])
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                # Latest data: calculate from last time point of entire data file
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = make_us_market_timestamps(df['timestamps'].iloc[-1], pred_len, time_diff)
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })

        trade_overlay = compute_trade_overlay(x_df, pred_df, actual_df)
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'trade_overlay': trade_overlay,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=7070)
