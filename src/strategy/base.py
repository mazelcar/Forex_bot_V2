"""Base Strategy Module for Forex Trading Bot V2.

This module provides the abstract base class for implementing trading strategies.
It defines the interface that all concrete strategies must implement and provides
common utility methods for strategy implementation.

The base strategy handles:
1. Configuration management
2. Market data validation
3. Signal generation interface
4. Risk management framework
5. Performance tracking

Author: mazelcar
Created: December 2024
"""

from abc import ABC, abstractmethod
import json
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    This class serves as a template for creating trading strategies. Any new strategy
    must inherit from this class and implement its required methods.

    Attributes:
        config_path (str): Path to the strategy's JSON configuration file
        config (Dict): Dictionary containing the strategy's settings loaded from config_path

    Example of creating a new strategy:
        class MovingAverageStrategy(Strategy):
            def __init__(self):
                super().__init__(config_file="strategies/ma_config.json")

            def _analyze_market_condition(self, data):
                # Implement your market analysis here
                pass

    Configuration File Structure:
        The JSON configuration file should contain:
        - strategy_name: Name of the strategy
        - market_scope: Trading pairs and timeframes
        - indicators: Technical indicators used
        - signal_conditions: Entry/exit conditions
        - filters: Additional trading filters
        - performance_metrics: Strategy performance settings

    Note:
        ABC (Abstract Base Class) means this is a template class and cannot be
        used directly. It must be inherited by concrete strategy classes that
        implement all required methods.
    """

    def __init__(self, config_file: str):
        """Initialize strategy with configuration.

        Args:
            config_file: Path to strategy configuration JSON file
        """
        self.config_path = Path(config_file)
        self.config = self._load_config()
        self.name = self.config.get('strategy_name', 'UnnamedStrategy')
        self.timeframes = self.config.get('market_scope', {}).get('timeframes', [])
        self.symbols = self.config.get('market_scope', {}).get('pairs', [])

        # Current market state
        self.current_market_condition: Dict = {}
        self.current_volatility: float = 0.0
        self.current_signal_strength: float = 0.0

        # Performance tracking
        self.trades_history: List[Dict] = []
        self.performance_metrics: Dict = {}

    def _load_config(self) -> Dict:
        """
        Load strategy configuration from a JSON file.
        This function only reads our strategy settings from the JSON file
        and converts it into a Python dictionary. The validation is handled
        separately by _validate_config().

        Returns:
            Dict: A dictionary containing all our strategy settings
        """
        try:
            # Step 1: Open and read our JSON configuration file
            # self.config_path is the location of our JSON file (passed when creating the strategy)
            # 'r' means we're opening it in read mode (just reading, not changing the file)
            # encoding='utf-8' tells Python how to read the text characters
            with open(self.config_path, 'r', encoding='utf-8') as f:
                # Step 2: Parse JSON into a Python dictionary
                # json.load() reads our JSON file and converts it into a Python dictionary
                # that we can easily work with in our code
                config = json.load(f)

            # Step 3: Check if our configuration is valid
            # This calls another function (_validate_config) that checks if our config
            # has all the required sections like 'strategy_name', 'market_scope', etc.
            # If something is missing, _validate_config will raise an error
            self._validate_config(config)

            # Step 4: If all checks pass, return our configuration
            # This config will be stored in self.config and used by other parts
            # of our strategy
            return config

        except Exception as e:
            # If anything goes wrong during this process (file not found, invalid JSON format,
            # missing required settings, etc.), create a new error with a clear message
            # The f"..." lets us include the original error message (e) in our new error
            raise ValueError(f"Error loading strategy config: {e}")

    def _validate_config(self, config: Dict) -> None:
        """
        Validate that the strategy configuration has all required sections and proper structure.
        This function checks if all mandatory sections exist in the config dictionary
        and validates specific sections like market_scope for required fields.

        Args:
            config: Dictionary containing the strategy configuration settings
                loaded from the JSON file

        Raises:
            ValueError: If any required section is missing or if sections are incomplete

        Required Sections:
            - strategy_name: Name identifier for the strategy
            - market_scope: Trading pairs and timeframes
            - indicators: Technical indicators configuration
            - signal_conditions: Entry/exit rules
            - filters: Risk management filters
            - performance_metrics: Performance measurement settings
        """
        # Step 1: Define all the sections that must be present in our config
        # These are the core components every strategy needs to function
        required_sections = [
            'strategy_name',
            'market_scope',
            'indicators',
            'signal_conditions',
            'filters',
            'performance_metrics'
        ]

        # Step 2: Check if each required section exists in our config
        # Loop through each required section and verify it's in the config
        for section in required_sections:
            if section not in config:
                # If any section is missing, raise an error with a clear message
                raise ValueError(f"Missing required config section: {section}")

        # Step 3: Validate the market_scope section specifically
        # Get the market_scope section, if it doesn't exist, get empty dict
        market_scope = config.get('market_scope', {})
        # Check if both 'pairs' and 'timeframes' exist and have values
        if not market_scope.get('pairs') or not market_scope.get('timeframes'):
            raise ValueError("Market scope must specify pairs and timeframes")

    def update_market_condition(self, data: pd.DataFrame) -> None:
        """
        Update the strategy's assessment of current market conditions.
        This function processes new market data to update our understanding
        of market conditions and adjust the strategy accordingly.

        Args:
            data: pandas DataFrame containing market data with columns like
                'open', 'high', 'low', 'close', 'volume', etc.

        Example market data format:
            DataFrame with columns:
            - timestamp: Time of the candle
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume
        """
        # Step 1: Analyze current market conditions
        # Send market data to _analyze_market_condition function to get market state
        self.current_market_condition = self._analyze_market_condition(data)

        # Step 2: Calculate current market volatility
        # Determine how volatile the market is using _calculate_volatility function
        self.current_volatility = self._calculate_volatility(data)

        # Step 3: Adjust strategy parameters
        # Based on new market conditions and volatility, adjust strategy settings
        self._adjust_parameters()

    @abstractmethod
    def _analyze_market_condition(self, data: pd.DataFrame) -> Dict:
        """
        Analyze current market conditions based on provided market data.
        This is an abstract method that must be implemented by any strategy class
        that inherits from this base class.

        Args:
            data: pandas DataFrame containing market data with required columns:
                - timestamp: Date and time of each candle
                - open: Opening price of the period
                - high: Highest price of the period
                - low: Lowest price of the period
                - close: Closing price of the period
                - volume: Trading volume of the period

        Returns:
            Dictionary containing the market condition assessment. For example:
            {
                'trend': 'uptrend',  # Current market trend
                'strength': 0.75,    # Trend strength (0 to 1)
                'volatility': 'high', # Market volatility assessment
                'support': 1.2345,   # Nearest support level
                'resistance': 1.2456 # Nearest resistance level
            }

        Example Implementation:
            def _analyze_market_condition(self, data: pd.DataFrame) -> Dict:
                # Calculate 20-period moving average
                ma20 = data['close'].rolling(20).mean()

                # Determine trend based on price vs moving average
                current_price = data['close'].iloc[-1]
                trend = 'uptrend' if current_price > ma20.iloc[-1] else 'downtrend'

                return {
                    'trend': trend,
                    'strength': 0.8,
                    'volatility': 'medium'
                }

        Note:
            - This is an abstract method (marked with @abstractmethod)
            - Any class inheriting from Strategy MUST implement this method
            - The method should contain your strategy's market analysis logic
            - The returned dictionary structure should be consistent with your strategy's needs
            - The analysis can include any technical indicators or custom calculations

        Raises:
            NotImplementedError: If the child class doesn't implement this method
        """
        pass

    @abstractmethod
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Calculate the current market volatility based on historical price data.
        This is an abstract method that must be implemented by any strategy class
        that inherits from this base class.

        Args:
            data: pandas DataFrame containing market data with required columns:
                - timestamp: Date and time of each candle
                - close: Closing prices for volatility calculation
                - high (optional): Highest prices, used for some volatility methods
                - low (optional): Lowest prices, used for some volatility methods

        Returns:
            float: A numerical measure of market volatility. For example:
                - 0.02 might represent 2% average price movement
                - Higher values indicate more volatile markets
                - Lower values indicate more stable markets

        Example Implementation:
            def _calculate_volatility(self, data: pd.DataFrame) -> float:
                # Calculate daily returns
                returns = data['close'].pct_change()

                # Calculate standard deviation of returns
                # Using 20-day rolling window
                volatility = returns.rolling(window=20).std()

                # Return the most recent volatility value
                return float(volatility.iloc[-1])

        Common Volatility Measures:
            1. Standard Deviation of Returns
            2. Average True Range (ATR)
            3. Bollinger Bands Width
            4. Historical Volatility
            5. Parkinson's Volatility

        Note:
            - This is an abstract method that must be implemented
            - The chosen volatility measure should align with your strategy's timeframe
            - Consider using a rolling window for the calculation
            - Handle missing data and zero values appropriately
            - Consider market-specific characteristics when choosing the method

        Raises:
            NotImplementedError: If the child class doesn't implement this method
            ValueError: If the data doesn't contain required price information
        """
        pass

    @abstractmethod
    def _adjust_parameters(self) -> None:
        """Adjust strategy parameters based on market conditions."""
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on strategy rules.

        Args:
            data: Market data for signal generation

        Returns:
            Dictionary containing signal information
        """
        pass

    @abstractmethod
    def calculate_position_size(self, account_info: Dict) -> float:
        """Calculate appropriate position size.

        Args:
            account_info: Current account information

        Returns:
            Position size in lots
        """
        pass

    @abstractmethod
    def calculate_stop_loss(self, data: pd.DataFrame, signal: Dict) -> Optional[float]:
        """Calculate stop loss price for a trade.

        Args:
            data: Market data for SL calculation
            signal: Signal information

        Returns:
            Stop loss price or None
        """
        pass

    @abstractmethod
    def calculate_take_profit(self, data: pd.DataFrame, signal: Dict) -> Optional[float]:
        """Calculate take profit price for a trade.

        Args:
            data: Market data for TP calculation
            signal: Signal information

        Returns:
            Take profit price or None
        """
        pass

    def validate_signal(self, signal: Dict, market_data: Union[Dict, pd.Series]) -> bool:
        """Validate a trading signal against current market conditions.

        Args:
            signal: Dictionary containing trading signal details
            market_data: Market data, either as dictionary or pandas Series

        Returns:
            bool: True if signal is valid, False otherwise
        """
        try:
            # Basic validation
            if not signal or market_data is None:
                return False

            # Convert market_data to dictionary if it's a Series
            market_dict = market_data if isinstance(market_data, dict) else market_data.to_dict()

            # Spread validation
            max_spread = self.config['filters']['spread']['max_spread_pips']
            current_spread = float(market_dict.get('spread', float('inf')))  # Convert to float

            if current_spread > max_spread:
                return False

            # Market session validation
            if not self._is_valid_session(market_dict):
                return False

            # Signal strength validation
            min_strength = 0.5  # Can be configured
            signal_strength = float(signal.get('strength', 0))  # Convert to float

            if signal_strength < min_strength:
                return False

            return True

        except Exception as e:
            print(f"Signal validation error: {e}")
            return False


    def _is_valid_session(self, market_data: Dict) -> bool:
        """Check if current market session is valid for trading.

        Args:
            market_data: Current market information

        Returns:
            True if session is valid for trading
        """
        # Implementation depends on specific requirements
        # This is a placeholder that should be overridden
        return True

    def update_trade_history(self, trade: Dict) -> None:
        """Update strategy's trade history.

        Args:
            trade: Completed trade information
        """
        self.trades_history.append(trade)
        self._update_performance_metrics()

    def _update_performance_metrics(self) -> None:
        """Update strategy performance metrics."""
        if not self.trades_history:
            return

        # Calculate basic metrics
        total_trades = len(self.trades_history)
        profitable_trades = len([t for t in self.trades_history if t.get('profit', 0) > 0])

        self.performance_metrics = {
            'total_trades': total_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'profit_factor': self._calculate_profit_factor(),
            'max_drawdown': self._calculate_max_drawdown()
        }

    def _calculate_profit_factor(self) -> float:
        """Calculate strategy profit factor."""
        gross_profit = sum(t.get('profit', 0) for t in self.trades_history if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in self.trades_history if t.get('profit', 0) < 0))

        return gross_profit / gross_loss if gross_loss != 0 else 0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.trades_history:
            return 0.0

        equity_curve = []
        peak = 0
        max_dd = 0

        for trade in self.trades_history:
            equity = sum(t.get('profit', 0) for t in self.trades_history[:len(equity_curve) + 1])
            equity_curve.append(equity)

            if equity > peak:
                peak = equity

            dd = (peak - equity) / peak if peak != 0 else 0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def get_info(self) -> Dict:
        """Get strategy information and current state.

        Returns:
            Dictionary containing strategy information
        """
        return {
            'name': self.name,
            'market_scope': {
                'symbols': self.symbols,
                'timeframes': self.timeframes
            },
            'current_condition': self.current_market_condition,
            'volatility': self.current_volatility,
            'signal_strength': self.current_signal_strength,
            'performance': self.performance_metrics
        }

    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name} - Active on {', '.join(self.symbols)}"