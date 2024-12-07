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

import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, List, Optional


class MT5Handler:
    """Handles MetaTrader 5 operations."""

    def __init__(self, debug: bool = False):
        """Initialize MT5 handler.

        Args:
            debug: Enable debug logging
        """
        self.connected = False
        self._initialize_mt5()

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