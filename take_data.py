import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json

# ==================== LOGGING SETUP ====================
def setup_logging(log_file: str = "binance_fetch.log"):
    """Configure logging for data fetch process"""
    logger = logging.getLogger("BinanceFetcher")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

# ==================== RATE LIMIT CALCULATOR ====================
class RateLimitManager:
    """Manage rate limit by weight of Binance Futures API"""
    
    # Default weights for endpoints
    WEIGHTS = {
        'klines': {1: 1, 5: 1, 10: 1, 20: 1, 50: 1, 100: 1, 500: 5, 1000: 10},
        'funding_rate': 1,  # For fetching funding rate history
    }
    
    # Rate limit restrictions (80% of 2400 for safety)
    MAX_WEIGHT_PER_MINUTE = 1920
    MINUTES_WINDOW = 1
    
    def __init__(self):
        self.request_weights = []
        self.request_times = []
        self.last_request_time = 0
    
    def calculate_weight(self, limit: int, endpoint: str = 'klines') -> int:
        """Calculate weight of request based on limit"""
        if endpoint == 'klines':
            weights = self.WEIGHTS['klines']
            return weights.get(limit, 10)
        elif endpoint == 'funding_rate':
            return self.WEIGHTS['funding_rate']
        return 1
    
    def get_wait_time(self, weight: int) -> float:
        """Calculate wait time (seconds) based on weight"""
        # Formula: (weight / 1920) * 60 seconds
        wait_seconds = (weight / self.MAX_WEIGHT_PER_MINUTE) * 60
        return wait_seconds
    
    def should_wait(self, weight: int) -> Tuple[bool, float]:
        """Check if should wait before sending request"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        self.request_weights = [
            w for t, w in zip(self.request_times, self.request_weights)
            if t > cutoff_time
        ]
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Calculate total weight in 1-minute window
        total_weight = sum(self.request_weights)
        
        if total_weight + weight > self.MAX_WEIGHT_PER_MINUTE:
            wait_time = self.get_wait_time(weight)
            return True, wait_time
        
        return False, 0
    
    def record_request(self, weight: int):
        """Record request for rate limit tracking"""
        self.request_times.append(time.time())
        self.request_weights.append(weight)
        self.last_request_time = time.time()

# ==================== DATA MERGER ====================
class DataMerger:
    """Merge OHLCV and Funding Rate data"""
    
    @staticmethod
    def merge_ohlcv_funding(
        ohlcv_df: pd.DataFrame,
        funding_df: pd.DataFrame,
        method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Merge OHLCV and Funding Rate data
        
        Args:
            ohlcv_df: OHLCV DataFrame with 'timestamp' column
            funding_df: Funding Rate DataFrame with 'timestamp' column
            method: Fill method ('ffill', 'bfill', 'interpolate')
        
        Returns:
            Merged DataFrame with columns: timestamp, open, high, low, close, volume, fundingRate
        """
        if ohlcv_df.empty:
            logger.warning("OHLCV DataFrame is empty, skipping merge")
            return ohlcv_df
        
        if funding_df.empty:
            logger.warning("Funding DataFrame is empty, adding fundingRate column with zeros")
            ohlcv_df['fundingRate'] = 0.0
            return ohlcv_df
        
        # Convert timestamp to datetime if not already
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'])
        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'])
        
        # Sort by timestamp
        ohlcv_df = ohlcv_df.sort_values('timestamp').reset_index(drop=True)
        funding_df = funding_df.sort_values('timestamp').reset_index(drop=True)
        
        # Merge using merge_asof (backward direction)
        merged_df = pd.merge_asof(
            ohlcv_df,
            funding_df,
            on='timestamp',
            direction='backward',  # Get nearest funding rate before timestamp
            tolerance=pd.Timedelta('8 hours')  # Funding rate updates every 8 hours
        )
        
        # Handle missing values
        if method == 'ffill':
            merged_df['fundingRate'] = merged_df['fundingRate'].fillna(method='ffill')
        elif method == 'bfill':
            merged_df['fundingRate'] = merged_df['fundingRate'].fillna(method='bfill')
        elif method == 'interpolate':
            merged_df['fundingRate'] = merged_df['fundingRate'].interpolate(method='linear')
        
        # Fill remaining NaN with 0
        merged_df['fundingRate'] = merged_df['fundingRate'].fillna(0.0)
        
        # Calculate coverage statistics
        non_zero_count = (merged_df['fundingRate'] != 0).sum()
        coverage_pct = (non_zero_count / len(merged_df) * 100) if len(merged_df) > 0 else 0
        
        logger.info(f"Successfully merged {len(merged_df)} rows")
        logger.info(f"Funding rate coverage: {coverage_pct:.2f}%")
        logger.info(f"Fill method used: {method}")
        
        return merged_df

# ==================== BINANCE FUTURES DATA FETCHER ====================
class BinanceFuturesFetcher:
    """Fetch OHLCV and Funding Rate data from Binance Futures"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Binance client
        
        Args:
            api_key: API key (optional, can use public endpoints)
            api_secret: API secret (optional)
        """
        exchange_config = {
            'enableRateLimit': True,
            'rateLimit': 500,
            'options': {
                'defaultType': 'future',
                'fetchTradingFees': False,
            }
        }
        
        if api_key and api_secret:
            exchange_config['apiKey'] = api_key
            exchange_config['secret'] = api_secret
        
        self.exchange = ccxt.binance(exchange_config)
        self.rate_limit_manager = RateLimitManager()
        self.data_merger = DataMerger()
        self.fetch_history = []
        self.ohlcv_data = None
        self.funding_data = None
        self.merged_data = None
    
    def get_symbol_pair(self, symbol: str) -> str:
        """Convert symbol to Binance Futures format (e.g., SOL/USDT)"""
        if '/' not in symbol:
            return f"{symbol}/USDT"
        return symbol
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '5m',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance Futures
        
        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')
            start_date: Start date
            end_date: End date
            timeframe: Time interval (e.g., '5m', '1h', '1d')
            limit: Max number of candles per request (1-1500)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        symbol = self.get_symbol_pair(symbol)
        limit = min(max(limit, 1), 1500)
        
        all_candles = []
        current_time = start_date
        
        # Calculate number of requests needed
        timeframe_ms = self._timeframe_to_ms(timeframe)
        total_ms = (end_date - start_date).total_seconds() * 1000
        total_requests = int(total_ms / (timeframe_ms * limit)) + 1
        
        logger.info(f"Fetching {symbol} {timeframe} from {start_date} to {end_date}")
        logger.info(f"Estimated requests: {total_requests}")
        
        pbar = tqdm(
            total=total_requests,
            desc=f"Fetching OHLCV {symbol}",
            unit="request",
            ncols=100
        )
        
        try:
            while current_time < end_date:
                try:
                    # Calculate weight
                    weight = self.rate_limit_manager.calculate_weight(limit)
                    
                    logger.debug(f"Request weight: {weight}")
                    
                    # Fetch data
                    candles = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since=int(current_time.timestamp() * 1000),
                        limit=limit
                    )
                    
                    if not candles:
                        break
                    
                    all_candles.extend(candles)
                    
                    # Update time for next request
                    last_candle_time = candles[-1][0]
                    current_time = datetime.fromtimestamp(last_candle_time / 1000)
                    
                    # Record request
                    self.rate_limit_manager.record_request(weight)
                    self.fetch_history.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'type': 'ohlcv',
                        'timeframe': timeframe,
                        'candles_fetched': len(candles),
                        'weight': weight
                    })
                    
                    pbar.update(1)
                    
                    if current_time >= end_date:
                        break
                
                except ccxt.NetworkError as e:
                    logger.error(f"Network error: {e}. Retrying in 5s...")
                    time.sleep(5)
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error: {e}. Stopping fetch for {symbol}")
                    break
        
        finally:
            pbar.close()
        
        # Convert to DataFrame
        if all_candles:
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Filter data in requested date range
            df = df[
                (df['timestamp'] >= start_date) &
                (df['timestamp'] <= end_date)
            ].reset_index(drop=True)
            
            self.ohlcv_data = df
            
            logger.info(
                f"Successfully fetched {len(df)} candles for {symbol} "
                f"({df['timestamp'].min()} to {df['timestamp'].max()})"
            )
            
            return df
        else:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()
    
    def fetch_funding_rates(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch Funding Rate data from Binance Futures
        
        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')
            start_date: Start date
            end_date: End date
            limit: Max number of records per request
        
        Returns:
            DataFrame with columns: timestamp, fundingRate
        """
        symbol = self.get_symbol_pair(symbol)
        limit = min(max(limit, 1), 1000)
        
        all_rates = []
        current_time = start_date
        
        # Calculate number of requests needed
        total_hours = (end_date - start_date).total_seconds() / 3600
        total_requests = int(total_hours / 3) + 1
        
        logger.info(f"Fetching funding rates for {symbol} from {start_date} to {end_date}")
        logger.info(f"Estimated requests: {total_requests}")
        
        pbar = tqdm(
            total=total_requests,
            desc=f"Fetching Funding Rates {symbol}",
            unit="request",
            ncols=100
        )
        
        try:
            while current_time < end_date:
                try:
                    # Call funding rate history API
                    rates = self.exchange.fetch_funding_rate_history(
                        symbol,
                        since=int(current_time.timestamp() * 1000),
                        limit=limit
                    )
                    
                    if not rates:
                        break
                    
                    all_rates.extend(rates)
                    
                    # Update time
                    last_rate_time = rates[-1]['timestamp']
                    current_time = datetime.fromtimestamp(last_rate_time / 1000)
                    
                    # Record request
                    weight = self.rate_limit_manager.calculate_weight(
                        limit,
                        endpoint='funding_rate'
                    )
                    self.rate_limit_manager.record_request(weight)
                    self.fetch_history.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'type': 'funding_rate',
                        'rates_fetched': len(rates),
                        'weight': weight
                    })
                    
                    pbar.update(1)
                    
                    if current_time >= end_date:
                        break
                
                except ccxt.NetworkError as e:
                    logger.error(f"Network error: {e}. Retrying in 5s...")
                    time.sleep(5)
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error: {e}. Stopping fetch")
                    break
        
        finally:
            pbar.close()
        
        # Convert to DataFrame
        if all_rates:
            df = pd.DataFrame([
                {
                    'timestamp': datetime.fromtimestamp(r['timestamp'] / 1000),
                    'fundingRate': r['fundingRate']
                }
                for r in all_rates
            ])
            
            # Filter data in requested date range
            df = df[
                (df['timestamp'] >= start_date) &
                (df['timestamp'] <= end_date)
            ].reset_index(drop=True)
            
            # Remove duplicates, keep last
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            self.funding_data = df
            
            logger.info(
                f"Successfully fetched {len(df)} funding rates for {symbol}"
            )
            
            return df
        else:
            logger.warning(f"No funding rate data fetched for {symbol}")
            return pd.DataFrame()
    
    def merge_data(self, method: str = 'ffill') -> pd.DataFrame:
        """
        Merge OHLCV and Funding Rate data
        
        Args:
            method: Fill method ('ffill', 'bfill', 'interpolate')
        
        Returns:
            Merged DataFrame
        """
        if self.ohlcv_data is None or self.ohlcv_data.empty:
            logger.error("OHLCV data is not available for merge")
            return pd.DataFrame()
        
        if self.funding_data is None or self.funding_data.empty:
            logger.warning("Funding data is not available, using OHLCV only")
            self.merged_data = self.ohlcv_data.copy()
            self.merged_data['fundingRate'] = 0.0
            return self.merged_data
        
        self.merged_data = self.data_merger.merge_ohlcv_funding(
            self.ohlcv_data,
            self.funding_data,
            method=method
        )
        
        return self.merged_data
    
    def save_merged_data(
        self,
        filepath: str = "merged_data.csv"
    ) -> str:
        """
        Save merged data to CSV file
        
        Args:
            filepath: Output CSV file path
        
        Returns:
            Path of saved file
        """
        if self.merged_data is None or self.merged_data.empty:
            logger.error("No merged data to save")
            return ""
        
        df = self.merged_data.copy()
        
        # Ensure correct column order
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'fundingRate']
        df = df[columns]
        
        # Format timestamp
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Round numeric columns
        df['open'] = df['open'].round(8)
        df['high'] = df['high'].round(8)
        df['low'] = df['low'].round(8)
        df['close'] = df['close'].round(8)
        df['volume'] = df['volume'].round(8)
        df['fundingRate'] = df['fundingRate'].round(10)
        
        # Save file
        df.to_csv(filepath, index=False)
        
        logger.info(f"Merged data saved to {filepath}")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Columns: {', '.join(columns)}")
        
        return filepath
    
    def fetch_and_merge(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '5m',
        limit: int = 1000,
        output_file: Optional[str] = None,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV, Funding Rate, merge, and save in one command
        
        Args:
            symbol: Trading pair
            start_date: Start date
            end_date: End date
            timeframe: OHLCV timeframe
            limit: Limit for OHLCV
            output_file: Output file path (if None, use default)
            fill_method: Fill method for missing values
        
        Returns:
            Merged DataFrame
        """
        logger.info("=" * 100)
        logger.info(f"Starting fetch and merge process for {symbol}")
        logger.info("=" * 100)
        
        # Step 1: Fetch OHLCV
        logger.info("\n[STEP 1/4] Fetching OHLCV data...")
        self.fetch_ohlcv(symbol, start_date, end_date, timeframe, limit)
        
        # Step 2: Fetch Funding Rate
        logger.info("\n[STEP 2/4] Fetching Funding Rate data...")
        self.fetch_funding_rates(symbol, start_date, end_date, limit=100)
        
        # Step 3: Merge
        logger.info("\n[STEP 3/4] Merging data...")
        self.merge_data(method=fill_method)
        
        # Step 4: Save
        logger.info("\n[STEP 4/4] Saving merged data...")
        if output_file is None:
            symbol_clean = symbol.replace('/', '')
            output_file = f"data/{symbol_clean}_{timeframe}_merged_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        # Create data directory if not exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.save_merged_data(output_file)
        
        logger.info("\n" + "=" * 100)
        logger.info("Fetch and merge process completed successfully!")
        logger.info("=" * 100)
        
        return self.merged_data
    
    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> float:
        """Convert timeframe to milliseconds"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        return timeframe_map.get(timeframe, 5 * 60 * 1000)
    
    def get_fetch_stats(self) -> Dict:
        """Get statistics about requests made"""
        if not self.fetch_history:
            return {}
        
        total_weight = sum(h.get('weight', 0) for h in self.fetch_history)
        total_requests = len(self.fetch_history)
        total_candles = sum(h.get('candles_fetched', 0) for h in self.fetch_history)
        total_rates = sum(h.get('rates_fetched', 0) for h in self.fetch_history)
        
        return {
            'total_requests': total_requests,
            'total_weight': total_weight,
            'average_weight_per_request': total_weight / total_requests if total_requests > 0 else 0,
            'total_candles': total_candles,
            'total_funding_rates': total_rates,
            'merged_rows': len(self.merged_data) if self.merged_data is not None else 0,
        }
    
    def save_fetch_history(self, filepath: str = "fetch_history.json"):
        """Save fetch history to JSON file"""
        history = [
            {
                **h,
                'timestamp': h['timestamp'].isoformat()
            }
            for h in self.fetch_history
        ]
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Fetch history saved to {filepath}")

# ==================== MAIN USAGE ====================
def main():
    """Example usage"""
    
    # Initialize fetcher
    fetcher = BinanceFuturesFetcher()
    
    # Set date/time
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 1, 10)
    
    # Fetch, merge, and save in one command
    merged_df = fetcher.fetch_and_merge(
        symbol='DOGE/USDT',
        start_date=start_date,
        end_date=end_date,
        timeframe='5m',
        limit=1000,
        output_file='data/DOGEUSDT_5m_merged_20250101_20250110.csv',
        fill_method='ffill'
    )
    
    # Display results
    if not merged_df.empty:
        print("\n" + "=" * 100)
        print("MERGED DATA PREVIEW:")
        print("=" * 100)
        print(merged_df.head(10))
        print(f"\nShape: {merged_df.shape}")
        print(f"Columns: {list(merged_df.columns)}")
        print(f"Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
        print(f"\nFunding Rate Statistics:")
        print(f"  Min: {merged_df['fundingRate'].min():.10f}")
        print(f"  Max: {merged_df['fundingRate'].max():.10f}")
        print(f"  Mean: {merged_df['fundingRate'].mean():.10f}")
    
    # Display statistics
    stats = fetcher.get_fetch_stats()
    print("\n" + "=" * 100)
    print("FETCH STATISTICS:")
    print("=" * 100)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:35s}: {value:,.2f}")
        else:
            print(f"{key:35s}: {value:,}")
    
    # Save fetch history
    fetcher.save_fetch_history()

if __name__ == '__main__':
    main()