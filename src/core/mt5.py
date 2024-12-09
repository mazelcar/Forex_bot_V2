"""MT5 Integration Module for Forex Trading Bot V2.

Handles core MetaTrader5 functionality:
1. Connection to MT5 terminal
2. Account information
3. Placing trades
4. Getting market data
5. Managing positions

Author: mazelcar
Created: December 2024
"""
import pandas as pd
import json
from pathlib import Path
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from src.core.market_sessions import MarketSessionManager


def create_default_config() -> bool:
    """Create default config directory and files if they don't exist."""
    try:
        # Setup config directory
        project_root = Path(__file__).parent.parent.parent.absolute()  # Go up to project root
        config_dir = project_root / "config"
        config_dir.mkdir(exist_ok=True)

        # Default MT5 config
        mt5_config_file = config_dir / "mt5_config.json"
        if not mt5_config_file.exists():
            default_config = {
                "username": "",
                "password": "",
                "server": "",
                "debug": True
            }

            with open(mt5_config_file, mode='w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)

            print(f"\nCreated default MT5 config file at: {mt5_config_file}")
            print("Please fill in your MT5 credentials in the config file")
            return False
        return True
    except Exception as e:
        print(f"Error creating config: {e}")
        return False


def get_mt5_config() -> Dict:
    """Get MT5 configuration from config file."""
    project_root = Path(__file__).parent.parent.parent.absolute()
    config_file = project_root / "config" / "mt5_config.json"

    if not config_file.exists():
        create_default_config()
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error reading config: {e}")
        return {}


class MT5Handler:
    """Handles MetaTrader 5 operations."""

    TIMEFRAME_MAPPING = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }

    def __init__(self, debug: bool = False):
        """Initialize MT5 handler.

        Args:
            debug: Enable debug logging
        """
        self.connected = False
        self.config = get_mt5_config()
        self._initialize_mt5()
        self.session_manager = MarketSessionManager()

    def _initialize_mt5(self) -> bool:
        """Connect to MT5 terminal."""
        try:
            if not mt5.initialize():
                return False
            self.connected = True
            return True
        except Exception as e:
            print(f"MT5 initialization error: {e}")
            return False

    def login(self, username: str, password: str, server: str) -> bool:
        """Login to MT5 account.

        Args:
            username: MT5 account ID
            password: MT5 account password
            server: MT5 server name
        """
        if not self.connected:
            return False

        try:
            return mt5.login(
                login=int(username),
                password=password,
                server=server
            )
        except Exception as e:
            print(f"MT5 login error: {e}")
            return False

    def get_account_info(self) -> Dict:
        """Get current account information."""
        if not self.connected:
            return {}

        try:
            account = mt5.account_info()
            if account is None:
                return {}

            return {
                'balance': account.balance,
                'equity': account.equity,
                'profit': account.profit,
                'margin': account.margin,
                'margin_free': account.margin_free
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}

    def get_market_status(self) -> Dict:
        """Get market status considering both MT5 and session times."""
        session_status = self.session_manager.get_status()

        # Cross-reference with MT5 symbols
        try:
            for market, info in self.session_manager.sessions['sessions'].items():
                # Check if any pair for this session is tradeable
                pairs = info.get('pairs', [])
                market_tradeable = any(
                    mt5.symbol_info(pair).trade_mode != 0
                    for pair in pairs
                    if mt5.symbol_info(pair) is not None
                )
                # Market is only open if both session time and MT5 allow trading
                session_status['status'][market] &= market_tradeable

            session_status['overall_status'] = 'OPEN' if any(session_status['status'].values()) else 'CLOSED'

        except Exception as e:
            print(f"Error checking MT5 market status: {e}")

        return session_status

    def place_trade(self, symbol: str, order_type: str, volume: float,
                   sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
        """Place a trade order.

        Args:
            symbol: Trading symbol (e.g. 'EURUSD')
            order_type: 'BUY' or 'SELL'
            volume: Trade volume in lots
            sl: Stop loss price
            tp: Take profit price
        """
        if not self.connected:
            return False

        try:
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': volume,
                'type': mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                'price': mt5.symbol_info_tick(symbol).ask if order_type == 'BUY' else mt5.symbol_info_tick(symbol).bid,
                'sl': sl,
                'tp': tp,
                'deviation': 10,
                'magic': 234000,
                'comment': 'ForexBot trade',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }

            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE

        except Exception as e:
            print(f"Error placing trade: {e}")
            return False

    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        if not self.connected:
            return []

        try:
            positions = mt5.positions_get()
            if positions is None:
                return []

            return [{
                'ticket': p.ticket,
                'symbol': p.symbol,
                'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': p.volume,
                'price': p.price_open,
                'profit': p.profit,
                'sl': p.sl,
                'tp': p.tp
            } for p in positions]

        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        include_volume: bool = True
    ) -> Optional[pd.DataFrame]:
        """Get historical price data from MT5.

        Args:
            symbol: Trading symbol (e.g. 'EURUSD')
            timeframe: String timeframe ('M1', 'M5', etc.)
            start_date: Start date for historical data
            end_date: End date for historical data
            include_volume: Whether to include volume data

        Returns:
            DataFrame with historical data or None if error
        """
        if not self.connected:
            print("MT5 not connected")
            return None

        try:
            # Validate and get MT5 timeframe constant
            tf = self.TIMEFRAME_MAPPING.get(timeframe)
            if tf is None:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Get historical data
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
            if rates is None or len(rates) == 0:
                print(f"No historical data available for {symbol} from {start_date} to {end_date}")
                # Return empty DataFrame with correct columns
                return pd.DataFrame(columns=[
                    'time', 'open', 'high', 'low', 'close',
                    'tick_volume', 'spread', 'real_volume'
                ])

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Add symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Symbol info not available for {symbol}")

            # Add pip value and point
            df['pip_value'] = 0.0001 if symbol[-3:] != 'JPY' else 0.01
            df['point'] = symbol_info.point

            # Sort by time in ascending order
            df = df.sort_values('time').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Error getting historical data: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed symbol information.

        Args:
            symbol: Trading symbol (e.g. 'EURUSD')

        Returns:
            Dictionary with symbol information or None if error
        """
        if not self.connected:
            return None

        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None

            return {
                'spread': info.spread,
                'digits': info.digits,
                'trade_mode': info.trade_mode,
                'volume_min': info.volume_min,
                'point': info.point,
                'pip_value': 0.0001 if symbol[-3:] != 'JPY' else 0.01,
                'tick_size': info.trade_tick_size
            }

        except Exception as e:
            print(f"Error getting symbol info: {e}")
            return None

    def validate_data_availability(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[bool, str]:
        """Validate if historical data is available for the specified period.

        Args:
            symbol: Trading symbol
            timeframe: String timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (is_valid, message)
        """
        if not self.connected:
            return False, "MT5 not connected"

        try:
            # Check symbol existence
            if mt5.symbol_info(symbol) is None:
                return False, f"Symbol {symbol} not found"

            # Validate timeframe
            if timeframe not in self.TIMEFRAME_MAPPING:
                return False, f"Invalid timeframe: {timeframe}"

            # Check data availability
            tf = self.TIMEFRAME_MAPPING[timeframe]
            test_data = mt5.copy_rates_range(symbol, tf, start_date, start_date + timedelta(days=1))

            if test_data is None or len(test_data) == 0:
                return False, f"No data available for {symbol} at {timeframe}"

            return True, "Data available"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def close_position(self, ticket: int) -> bool:
        """Close a specific position.

        Args:
            ticket: Position ticket number
        """
        if not self.connected:
            return False

        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False

            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'position': ticket,
                'symbol': position[0].symbol,
                'volume': position[0].volume,
                'type': mt5.ORDER_TYPE_SELL if position[0].type == 0 else mt5.ORDER_TYPE_BUY,
                'price': mt5.symbol_info_tick(position[0].symbol).bid if position[0].type == 0 else mt5.symbol_info_tick(position[0].symbol).ask,
                'deviation': 10,
                'magic': 234000,
                'comment': 'ForexBot close',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }

            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE

        except Exception as e:
            print(f"Error closing position: {e}")
            return False

    def __del__(self):
        """Clean up MT5 connection."""
        if self.connected:
            mt5.shutdown()