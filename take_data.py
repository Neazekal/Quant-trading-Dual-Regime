import ccxt
import pandas as pd
from datetime import datetime
import time
import logging
from typing import Optional
from pathlib import Path
from tqdm import tqdm

# ==================== LOGGING SETUP ====================
logger = logging.getLogger("BinanceFetcher")
logger.addHandler(logging.NullHandler())
logger.disabled = True

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
        self.data_merger = DataMerger()
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
